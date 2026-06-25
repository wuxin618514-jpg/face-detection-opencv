import os #----------------------------------------------------------------------------------------------------------------------------5
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #------------------------------------------------------------------------------------------5

import torch            #1,3,4,5
import cv2              #5
import numpy as np      #4

from data.config import cfg_re50             #1，5
from models.retinaface import RetinaFace     #1，5
from utils.box_utils import decode           #4，5
from utils.py_cpu_nms import py_cpu_nms      #4，5

device = torch.device("cpu") #---------------------------------------------------------------------------------------------------------1
print(f"Loading model on execution device: {device}")

cfg = cfg_re50                                                                                                                        
net = RetinaFace(cfg=cfg, phase="test")
state_dict = torch.load("./weights/Resnet50_Final.pth", map_location=device)

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

net.load_state_dict(new_state_dict)
net.to(device)
net.eval()
print("CPU initialized successfully.") #-----------------------------------------------------------------------------------------------1

# ===================== Fixed Anchor Generation =====================
def make_priors(cfg): #----------------------------------------------------------------------------------------------------------------3
    h, w = 480, 640
    min_sizes = cfg['min_sizes']
    steps = cfg['steps']
    priors = []
    for k, step in enumerate(steps):
        for i in range(int(h // step)):
            for j in range(int(w // step)):
                for size in min_sizes[k]:
                    cx = (j + 0.5) * step / w
                    cy = (i + 0.5) * step / h
                    bw = size / w
                    bh = size / h
                    priors.append([cx, cy, bw, bh])
    return torch.tensor(priors, dtype=torch.float32)

priors = make_priors(cfg).to(device) #------------------------------------------------------------------------------------------------3

def auto_resize(img, target_w=640, target_h=480):#------------------------------------------------------------------------------------2
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    img_resize = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_pad = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    dw = (target_w - new_w) // 2
    dh = (target_h - new_h) // 2
    img_pad[dh:dh+new_h, dw:dw+new_w] = img_resize
    return img_pad, scale, dw, dh#-----------------------------------------------------------------------------------------------------2

def resize_to_fit_screen(img, max_w=1200, max_h=800):#---------------------------------------------------------------------------------5
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img#------------------------------------------------------------------------------------------------------------------------5

def detect_image():#-------------------------------------------------------------------------------------------------------------------5
    while True:
        print("\n------------------------------------------------")
        img_name = input("Enter image name/path (or type 'q' to return to menu): ").strip()
                
        if img_name.lower() == 'q':
            print("Returning to main menu...")
            break

        if not os.path.exists(img_name):
            print(f"[Error] Image not found in current directory: {img_name}")
            continue

        print(f"Processing image: [{img_name}]")
        img_raw = cv2.imread(img_name)
        if img_raw is None:
            print("[Error] Failed to read the image!")
            continue
            
        h_orig, w_orig = img_raw.shape[:2]
        t_w, t_h = 640, 480 
        img_pad, scale, dw, dh = auto_resize(img_raw, t_w, t_h)#-----------------------------------------------------------------------5

        img_data = img_pad.astype(np.float32) - np.array([104, 117, 123], dtype=np.float32)#-------------------------------------------2
        blob = torch.from_numpy(img_data).permute(2, 0, 1).unsqueeze(0).to(device).float()#--------------------------------------------2

        with torch.no_grad():#---------------------------------------------------------------------------------------------------------3
            loc, conf, _ = net(blob)#--------------------------------------------------------------------------------------------------3

        boxes = decode(loc.squeeze(0), priors, cfg['variance'])#-----------------------------------------------------------------------4
        scale_tensor = torch.tensor([t_w, t_h, t_w, t_h], dtype=torch.float32).to(device)
        boxes = boxes * scale_tensor
        scores = conf.squeeze(0)[:, 1]

        mask = scores > 0.40
        boxes = boxes[mask].cpu().numpy()
        scores = scores[mask].cpu().numpy()

        face_count = 0
        base_line = max(2, int(min(w_orig, h_orig) / 400))                           

        if len(boxes) > 0:
            dets = np.hstack((boxes, scores[:, np.newaxis]))
            keep = py_cpu_nms(dets, 0.3)
            face_count = len(keep)
            
            for i in keep:
                box = dets[i]
                x1 = max(0, int((box[0] - dw) / scale))
                y1 = max(0, int((box[1] - dh) / scale))
                x2 = min(w_orig, int((box[2] - dw) / scale))
                y2 = min(h_orig, int((box[3] - dh) / scale))
                cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 255, 0), base_line)#----------------------------------------------------4
                
        font_scale = 0.8#--------------------------------------------------------------------------------------------------------------5
        text_thickness = 2
        hud_text = f"Faces Counted: {face_count}"
        
        (text_w, text_h), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        pad_box = int(text_h * 0.6)
        
        overlay = img_raw.copy()
        x_start, y_start = 20, 20
        x_end = x_start + text_w + pad_box * 2
        y_end = y_start + text_h + pad_box * 2
        
        cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (0, 255, 255), 1)
        
        cv2.addWeighted(overlay, 0.25, img_raw, 0.75, 0, img_raw)
        
        cv2.putText(img_raw, hud_text, (x_start + pad_box, y_start + text_h + pad_box), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)

        img_show = resize_to_fit_screen(img_raw, max_w=1200, max_h=800)

        cv2.startWindowThread()
        cv2.namedWindow("Result Window", cv2.WINDOW_AUTOSIZE)
        
        cv2.imshow("Result Window", img_show)
        print("Window opened. Press ANY key on the image window to close it and enter next image.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()#-------------------------------------------------------------------------------------------------------5

def detect_camera():#------------------------------------------------------------------------------------------------------------------5
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera access failed.")
        return

    print("Camera stream initialized. Press 'q' on the camera window to return to menu.")
    t_w, t_h = 640, 480
    current_priors = priors
    scale_tensor = torch.tensor([t_w, t_h, t_w, t_h], dtype=torch.float32).to(device)

    cv2.startWindowThread()
    cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)

    prev_frame_time = time.time()
    avg_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_orig, w_orig = frame.shape[:2]
        img_pad, scale, dw, dh = auto_resize(frame, t_w, t_h)#-------------------------------------------------------------------------5

        frame_data = img_pad.astype(np.float32) - np.array([104, 117, 123], dtype=np.float32)#-----------------------------------------2
        frame_data = frame_data.transpose(2, 0, 1)
        blob = torch.from_numpy(frame_data).unsqueeze(0).to(device).float()#-----------------------------------------------------------2

        with torch.no_grad():#---------------------------------------------------------------------------------------------------------3
            loc, conf, _ = net(blob)#--------------------------------------------------------------------------------------------------3

        boxes = decode(loc.squeeze(0), current_priors, cfg['variance'])#---------------------------------------------------------------4
        boxes = boxes * scale_tensor
        scores = conf.squeeze(0)[:, 1]

        mask = scores > 0.50
        boxes = boxes[mask].cpu().numpy()
        scores = scores[mask].cpu().numpy()

        face_count = 0

        if len(boxes) > 0:
            dets = np.hstack((boxes, scores[:, np.newaxis]))
            keep = py_cpu_nms(dets, 0.3)
            face_count = len(keep)
            
            for i in keep:
                box = dets[i]
                x1 = max(0, int((box[0] - dw) / scale))
                y1 = max(0, int((box[1] - dh) / scale))
                x2 = min(w_orig, int((box[2] - dw) / scale))
                y2 = min(h_orig, int((box[3] - dh) / scale))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)#--------------------------------------------------------------4

        current_frame_time = time.time()#----------------------------------------------------------------------------------------------5
        time_delta = current_frame_time - prev_frame_time
        prev_frame_time = current_frame_time
        if time_delta > 0:
            instant_fps = 1.0 / time_delta
            avg_fps = (0.9 * avg_fps) + (0.1 * instant_fps) if avg_fps != 0.0 else instant_fps

        camera_overlay = frame.copy()
        cv2.rectangle(camera_overlay, (15, 15), (185, 80), (20, 20, 20), -1)
        cv2.rectangle(camera_overlay, (15, 15), (185, 80), (100, 100, 100), 1)
        cv2.addWeighted(camera_overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {face_count}", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing camera stream...")
            break

    cap.release()
    cv2.destroyAllWindows()                                                                                    

if __name__ == "__main__":                                                                                     
    while True:
        print("\n================ MAIN MENU ================")
        print("1. Static Image Mode (Looping)")
        print("2. Live Camera Mode")
        print("3. Exit Program")
        print("===========================================")
        
        choice = input("Select execution mode (1, 2, or 3): ").strip()
        if choice == "1":
            detect_image()
        elif choice == "2":
            detect_camera()
        elif choice == "3":
            print("Exiting RetinaFace System. Goodbye!")
            break
        else:
            print("[Warning] Invalid choice! Please enter 1, 2, or 3.")#---------------------------------------------------------------5
            
#Corresponding code members:
#1 Xu Mingze
#2 Wang Xichen
#3 Wu Xin
#4 Wang Shenghao
#5 Yang Shuhan