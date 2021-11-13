import gdown
import os

url = ['https://drive.google.com/uc?id=1-CfTd4JnBvIKOojK1oeLIhzCzd4r5Ki7',
    'https://drive.google.com/uc?id=1-1gCA8u0stejgAUr3OGL-4bgzRHn2gUj',
    'https://drive.google.com/uc?id=1-C6IeWyENKPjQrbiECPgiqnuKxez5pfe']
output = ['yolov3.weights', 'yolov3.cfg', 'coco.names']

for i in range(len(url)):
    gdown.download(url[i], output[i], quiet=False)
