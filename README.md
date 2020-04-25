# PyTorch Object Detection and Tracking
Object detection in images, and tracking across video frames

### How to use
1. Run the download_weights.sh script in the config folder to download the Yolo weights file.
2. run(recommend do it in venv or pinenv):
```shell script pip3 install -r requirements.txt --user```
3. Then run next command: 
```shell script
python3 object_tracker.py -p <video_path>
```
There the -p means path to video including the file name and extension (.avi and .mp4 are tested). 

Full story at:
https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98

References:
1. YOLOv3: https://pjreddie.com/darknet/yolo/
2. Erik Lindernoren's YOLO implementation: https://github.com/eriklindernoren/PyTorch-YOLOv3
3. YOLO paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf
4. SORT paper: https://arxiv.org/pdf/1602.00763.pdf
5. Alex Bewley's SORT implementation: https://github.com/abewley/sort
6. Installing Python 3.6 and Torch 1.0: https://medium.com/@chrisfotache/getting-started-with-fastai-v1-the-easy-way-using-python-3-6-apt-and-pip-772386952d03
