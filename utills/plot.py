import numpy as np
import cv2

class plot:

    # Fungsi u/ melakukan transformasi bird-view
    def bird_eye_view(frame, distances_mat, bottom_points, scale_w, scale_h, risk_count):
        h = frame.shape[0]
        w = frame.shape[1]

        red = (0, 0, 255, 255)
        green = (0, 255, 0, 255)
        white = (200, 200, 200, 255)
        map_bg = (0,0,0,0)
        black = (0,0,0, 255)
        
        blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 4), np.uint8)
        blank_image[:] = white
        red_image = np.zeros((int(h * scale_h), int(w * scale_w), 4), np.uint8)
        red_image[:] = map_bg
        
        warped_pts = []
        r = []
        g = []

        for i in range(len(distances_mat)):
            if distances_mat[i][2] == 0:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g):
                    r.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g):
                    r.append(distances_mat[i][1])
                    
                # menggambar garis untuk pelanggaran    
                blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), 
                                                     int(distances_mat[i][0][1] * scale_h)), 
                                       (int(distances_mat[i][1][0] * scale_w), 
                                        int(distances_mat[i][1][1]* scale_h)), red, 2)

        for i in range(len(distances_mat)):
            if distances_mat[i][2] == 2:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g):
                    g.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g):
                    g.append(distances_mat[i][1])
        
        # menggambar circle pada setiap objek yg terdeteksi
        for i in bottom_points:
            blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, green, 10)
        for i in r:
            blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, red, 10)
            red_image = cv2.circle(red_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 10, red, 10)
        return blank_image, red_image
    
    
    
    # Fungsi u/ drawing bounding boxes pada frame perspective view--
    # --dan drawing lines antar objek yg melakukan pelanggaran
    def social_distancing_view(frame, distances_mat, boxes, risk_count, bird_view):

        red = (0, 0, 255)
        green = (0, 255, 0)
        
        # gambar bounding box u/ yg tidak melanggar
        for i in range(len(boxes)):
            x,y,w,h = boxes[i][:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),green,2)

        for i in range(len(distances_mat)):
            per1 = distances_mat[i][0]
            per2 = distances_mat[i][1]
            closeness = distances_mat[i][2]

            if closeness == 0:
                x,y,w,h = per1[:]
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),red,2)

                x1,y1,w1,h1 = per2[:]
                frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),red,2)

                frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 2)
                
            # buat pad (padding) pada sisi bawah frame prespective-view
            # dengan h=100, dan w=frame.shape[1]
            pad = np.full((100,frame.shape[1],4), [250, 250, 250, 255], dtype=np.uint8)

            # draw text pada padding
            cv2.putText(pad, "Jumlah Orang Terdeteksi : " + str(risk_count[0] + risk_count[1]) + " Orang", (100, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 0), 2)
            cv2.putText(pad, "Jumlah Pelanggaran Social Distancing : " + str(risk_count[0]) + " Orang", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
        # gabungkan pad dengan frame prespective-view
        # dan kemudian gabungkan dengan bird-view
        frame = np.vstack((frame,pad))
        frame = np.hstack((frame, bird_view))
    
        return frame