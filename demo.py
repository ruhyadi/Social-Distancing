import cv2
from utills.math import calculate_social_distancing
from utills import plot
from utills import utills
import argparse

parser = argparse.ArgumentParser(description='Social Distancing Detection')
parser.add_argument('--input_vid', type=str, default='demo/video.mp4', help='input video path')
parser.add_argument('--output', type=str, default='demo/output.avi', help='path to ouput.avi')
parser.add_argument('--confidence', type=int, default=0.5)
parser.add_argument('--threshold', type=int, default=0.5)
parser.add_argument('--weights', type=str, default='models/yolov3.weights', help='YOLOv3 weights')
parser.add_argument('--config', type=str, default='models/yolov3.cfg', help='YOLOv3 config')

args = parser.parse_args()

# initialize confidence dan threshold
confid = args.confidence
thresh = args.threshold

# initialize 7 titik transformasi prespective-view ke bird-view
# 4 titik pertama (bottom-left, bottom-right, top-right, top-left) digunakan u/--
# --melakukan transformasi prespective ke bird-view
# 3 titik setelahnya digunakan untuk menghitung pixel-to-metric ratio
pts = [(0, 236.2), (615.86, 346.92), (853.5, 7), (500.25, 7), 
       (382.11, 271.98), (445.91, 285.82), (427, 236.2)]

# Load Yolov3 weights
weightsPath = args.weights
configPath = args.config

net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net_yl.getLayerNames()
ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

net_yl.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net_yl.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

# initialize video path
video_path = args.input_vid

# hitung waktu running; MULAI
t1 = cv2.getTickCount()

# processing social distancing detector
calculate_social_distancing(video_path, 
    net_yl, 
    ln1=ln1, 
    points=pts, 
    confid=confid, 
    thresh=thresh,
    output=args.output)

# hitung waktu running; SELESAI
t2 = cv2.getTickCount()

# hitung waktu running
t3 = (t2 - t1)/ cv2.getTickFrequency()
print("Waktu yang dibutuhkan :", t3)