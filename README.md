# face-detection-opencv
Face Detection using Python and OpenCV

We provide restnet50 and mobilenet0.25 as backbone network to train model. We trained Mobilenet0.25 on imagenet dataset and get 46.58% in top 1. If you do not wish to train the model, we also provide trained model. Pretrain model and trained model are put in google cloud and baidu cloud Password: fstq . The model could be put as follows:

  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in data/config.py and train.py.

Train the model using WIDER FACE:

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25