# Social-Distancing
Bird-eye view social distancing detector with YOLOv3

<a href="https://www.kaggle.com/didiruh/proyek-kb-social-distancing"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>

## 1. Install Requirements
Install depedencies in requirements.
```
pip install -r requirements.txt
```

## 2. Download YOLOv3 Models
Downloa YOLOv3 models and save it to `models` directory.
```
cd models
python download_yolov3.py
```

## 3. Run Demo
Running demo with 
```
python demo.py \
    --input_vid=demo/video.mp4 \
    --output=demo/output.avi
```