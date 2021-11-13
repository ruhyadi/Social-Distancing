import numpy as np
import cv2
import utills
import plot
import time
import imutils


# Fungsi u/ melakukan processing social distancing detector
def calculate_social_distancing(vid_path, net, ln1, points, confid, thresh):
    
    # initialize count dan video capture
    count = 0
    vs = cv2.VideoCapture(vid_path)    

    # Ambil video height, width dan fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    
    # Tentukan skala untuk bird-view
    scale_w, scale_h = utills.get_scale(width, height)
    
    # initialize penyimpanan video output hasil processing
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    output_movie = cv2.VideoWriter("Social Distancing Detection.avi", fourcc, fps, (1540, 720), True)
    
    # variable berisi nilai red / map pelanggaran
    RED_IMAGE = []
    
    global image
    
    # mulai processing dengan loop pada video capture
    while True:
        
        # ambil grab dan frame
        (grabbed, frame) = vs.read()

        # berhenti saat nilai grab = false
        if not grabbed:
            print("[INFO] Processing done...")
            break
        
        # ambil H dan W dari frame vs
        (H, W) = frame.shape[:2]
          
        # initialize src yg berisi 4 titik tranformasi, dan--
        # --dst yang berisi 4 titik ukuran vs sebenarnya
        # prespective_transform berisi matriks u/ transformasi prespective ke bird-view
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)
        
        # invers matriks u/ transformasi bird ke prespective-view
        inv_trans = np.linalg.pinv(prespective_transform)
        
        # gunakan 3 titik setelahnya u/ pixel-to-metric ratio dalam variable pts
        # warped_pt berisi transformasi 3 titik pada bird-view
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
        
        # initialize distance_w, dan distance_h yg masing2 berisi jarak 180 cm dalam satuan pixel
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        
        # draw 4 titik transformasi pada frame video prespective-view
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    
        # Memproses deteksi dengan pre-trained model YOLO-v3
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        
        boxes = []
        confidences = []
        classIDs = []   
        rects = []
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                # deteksi (hanya) orang pada frame
                # YOLO menggunakan dataset COCO dimana index human dalam--
                # --dataset berada pada index 0
                if classID == 0:

                    if confidence > confid:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                                    
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
                
        if len(boxes1) == 0:
            count = count + 1
            continue
            
        # initialize bottom-center dari setiap bounding-box yg terdeteksi, dan--
        # initialize rects yg berisi bottom-center untuk digunakan pada generating ID's
        person_points, rects = utills.get_transformed_points(boxes1, prespective_transform)

        
        frame1 = np.copy(frame)
        # convert frame1 (rgb) to frame1 (rbga)
        r, g, b = cv2.split(frame1)
        alpha_frame1 = np.ones(r.shape, np.uint8)*255
        frame1 = cv2.merge((r,g,b,alpha_frame1))
        
        # Hitung dan pelanggaran jarak antar objek (manusia) pada transformasi bird-view
        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = utills.get_count(distances_mat)
    
        # Draw bird eye view and frame with bouding boxes around humans according to risk factor 
        # Hasilkan video output transformasi bird-view, dan--
        # gunakan hasil u/ digabungkan dgn video output prespective-view 
        bird_image, red_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, 
                                        risk_count)
        RED_IMAGE.append(red_image)
        
        # buat map pelanggaran
        for red_image in RED_IMAGE:
            red_point = red_image[:, :, 3] > 0
            bird_image[red_point] = red_image[red_point]
            
        img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count, bird_image)
        
        # resizing video output
        img = imutils.resize(img, height=720)
        img_r, img_g, img_b, img_a = cv2.split(img)
        img = cv2.merge((img_r,img_g,img_b))
        
        # write video
        if count != 0:
            output_movie.write(img)    
        count = count + 1