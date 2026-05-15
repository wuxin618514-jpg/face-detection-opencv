import cv2

def load_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def detect_faces_in_image(image_path, face_cascade):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Image Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_in_camera(face_cascade):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Camera Face Detection (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_cascade = load_face_detector()
    
    print("=== Face Detection Tool ===")
    print("1. Detect faces in an image")
    print("2. Detect faces in real-time camera feed")
    
    choice = input("Enter your choice (1/2): ")
    
    if choice == "1":
        path = input("Enter the image path (e.g., test.jpg): ")
        detect_faces_in_image(path, face_cascade)
    elif choice == "2":
        detect_faces_in_camera(face_cascade)
    else:
        print("Invalid choice. Exiting.")