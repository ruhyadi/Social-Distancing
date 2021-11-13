import numpy as np
import cv2 

class utills:

    # Fungsi u/ menentukan bottom center dari semua bounding boxes object--
    # --dan kemudian akan digunakan u/ melakukan transformasi dari prespective-view ke bird-view
    def get_transformed_points(boxes, prespective_transform):
        
        # initialize rects dan bottom_points yg berguna untuk menyimpan array bottom center--
        # --dari bounding box object
        rects = []
        bottom_points = []
        
        for box in boxes:
            pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+box[3])]]] , dtype="float32")
            bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
            pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
            pnt_bird = [int(bd_pnt[0]), int(bd_pnt[1]), 0, 0]
            
            bottom_points.append(pnt)
            rects.append(np.array(pnt_bird))

        return bottom_points, rects
    
    
    # Fungsi u/ menghitung jarak antar dua point object (orang).
    # distance_w, distance_h mempresentasikan pixel-to-metric ratio atau--
    # --besar nilai pixel untuk jarak 180 cm dalam frame video.
    def cal_dis(p1, p2, distance_w, distance_h):

        h = abs(p2[1]-p1[1])
        w = abs(p2[0]-p1[0])

        dis_w = float((w/distance_w)*180)
        dis_h = float((h/distance_h)*180)

        return int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))

    
    # Fungsi u/ menghitung jarak antar semua titik object dan--
    # --menghiutng closeness ratio (rasio kedekatan).
    def get_distances(boxes1, bottom_points, distance_w, distance_h):

        distance_mat = []
        bxs = []

        for i in range(len(bottom_points)):
            for j in range(len(bottom_points)):
                if i != j:
                    dist = utills.cal_dis(bottom_points[i], bottom_points[j], distance_w, distance_h)
                    if dist <= 180:
                        closeness = 0
                        distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                        bxs.append([boxes1[i], boxes1[j], closeness])      
                    else:
                        closeness = 2
                        distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                        bxs.append([boxes1[i], boxes1[j], closeness])

        return distance_mat, bxs

    
    # Function gives scale for birds eye view  
    # Fungsi memberikan skala untuk transformasi bird-view
    # Skala yg digunakan w:480, h:1180 (video=1080 + pad=100) 
    def get_scale(W, H):
        dis_w = 387
        dis_h = 580
        return float(dis_w/W),float(dis_h/H)

    
    # Fungsi u/ menghitung jumlah objek (orang) yg melakukan pelanggaran
    def get_count(distances_mat):
        r = []
        g = []
        for i in range(len(distances_mat)):
            if distances_mat[i][2] == 0:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g):
                    r.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g):
                    r.append(distances_mat[i][1])
        for i in range(len(distances_mat)):
            if distances_mat[i][2] == 2:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g):
                    g.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g):
                    g.append(distances_mat[i][1])
        return (len(r), len(g))