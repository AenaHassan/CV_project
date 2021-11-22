import cv2
import numpy as np
import time

def main():
    # Load Yolo
    net = cv2.dnn.readNet("mask-yolov3_10000.weights", "mask-yolov3.cfg")
    classes = []
    with open("mask-obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    print("Press 1 for pre-recorded videos, 2 for live stream: ")
    option = int(input())

    if option == 1:
        # Record video
        windowName_1= "Sample Feed from Camera 1"
        windowName_2= "Sample Feed from Camera 2"
        windowName_3= "Sample Feed from Camera 3"
        cv2.namedWindow(windowName_1)
        cv2.namedWindow(windowName_2)
        cv2.namedWindow(windowName_3)


        capture1 = cv2.VideoCapture("video1.avi")  # phone 1 camera
        capture2 = cv2.VideoCapture("video2.avi")   # laptop camera
        capture3 = cv2.VideoCapture("video3.avi") #phone 2 camera

        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id_1 = 0
        frame_id_2 = 0
        frame_id_3 = 0
        # define size for recorded video frame for video 1
        width1 = int(capture1.get(3))
        height1 = int(capture1.get(4))
        size1 = (width1, height1)

        # define size for recorded video frame for video 2
        width2 = int(capture2.get(3))
        height2 = int(capture2.get(4))
        size2 = (width2, height2)

        # define size for recorded video frame for video 3
        width3 = int(capture3.get(3))
        height3 = int(capture3.get(4))
        size3 = (width3, height3)


        
        # frame of size is being created and stored in .avi file
        optputFile1 = cv2.VideoWriter(
            'cam1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
        optputFile2 = cv2.VideoWriter(
            'cam2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size2)
        optputFile3 = cv2.VideoWriter(
            'cam3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size3)
        


        # check if feed exists or not for camera 1
        if capture1.isOpened():
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
        else:
            ret1 = False
            ret2 = False
            ret3 = False
            

        while ret1 and ret2 and ret3:
            ret1, frame1 = capture1.read()
            frame_id_1 += 1
            # Detecting objects
            blob_1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_1)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width1)
                        center_y = int(detection[1] * height1)
                        w = int(detection[2] * width1)
                        h = int(detection[3] * height1)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame1, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame1, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)



            elapsed_time = time.time() - starting_time
            fps = frame_id_1 / elapsed_time
            cv2.putText(frame1, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)

                    











            
            ret2, frame2 = capture2.read()
            frame_id_2 += 1
            # Detecting objects
            blob_2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_2)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width2)
                        center_y = int(detection[1] * height2)
                        w = int(detection[2] * width2)
                        h = int(detection[3] * height2)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame2, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame2, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)



            elapsed_time = time.time() - starting_time
            fps = frame_id_2 / elapsed_time
            cv2.putText(frame2, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)


            ret3, frame3 = capture3.read()
            frame_id_3 += 1
            # Detecting objects
            blob_3 = cv2.dnn.blobFromImage(frame3, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_3)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width3)
                        center_y = int(detection[1] * height3)
                        w = int(detection[2] * width3)
                        h = int(detection[3] * height3)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame3, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame3, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)



            elapsed_time = time.time() - starting_time
            fps = frame_id_3 / elapsed_time
            cv2.putText(frame3, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)


           
            # sample feed display from camera 1
            cv2.imshow(windowName_1, frame1)
            cv2.imshow(windowName_2, frame2)
            cv2.imshow(windowName_3, frame3)

            # saves the frame from camera 1
            optputFile1.write(frame1)
            optputFile2.write(frame2)
            optputFile3.write(frame3)

            # escape key (27) to exit
            if cv2.waitKey(1) == 27:
                break
        capture1.release()
        cv2.destroyAllWindows()

    elif option == 2:
        # live stream
        windowName1 = "Live Stream Camera 1"
        windowName2 = "Live Stream Camera 2"
        windowName3 = "Live Stream Camera 3"
        
        cv2.namedWindow(windowName1)
        cv2.namedWindow(windowName2)
        cv2.namedWindow(windowName3)



        

        frame_id_1 = 0
        frame_id_2 = 0
        frame_id_3 = 0

        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()



        capture1 = cv2.VideoCapture("http://10.104.6.247:8080/video")  # phone 1 camera
        capture2 = cv2.VideoCapture(0)   # laptop camera
        capture3 = cv2.VideoCapture("http://10.104.2.33:8080/video") #phone 2 camera

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
            
        else:
            ret1 = False
            ret2 = False
            ret3 = False



        # define size for recorded video frame for video 1
        width1 = int(capture1.get(3))
        height1 = int(capture1.get(4))
        size1 = (width1, height1)

        # define size for recorded video frame for video 2
        width2 = int(capture2.get(3))
        height2 = int(capture2.get(4))
        size2 = (width2, height2)

        # define size for recorded video frame for video 3
        width3 = int(capture3.get(3))
        height3 = int(capture3.get(4))
        size3 = (width3, height3)



        while ret1 and ret2 and ret3:
            ret1, frame1 = capture1.read()
            frame_id_1 += 1
            # Detecting objects
            blob_1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_1)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width1)
                        center_y = int(detection[1] * height1)
                        w = int(detection[2] * width1)
                        h = int(detection[3] * height1)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame1, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame1, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)



            elapsed_time = time.time() - starting_time
            fps = frame_id_1 / elapsed_time
            cv2.putText(frame1, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)


            
            cv2.imshow(windowName1, frame1)
            ret2, frame2 = capture2.read()






            frame_id_2 += 1
            # Detecting objects
            blob_2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_2)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width2)
                        center_y = int(detection[1] * height2)
                        w = int(detection[2] * width2)
                        h = int(detection[3] * height2)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame2, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame2, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)



            elapsed_time = time.time() - starting_time
            fps = frame_id_2 / elapsed_time
            cv2.putText(frame2, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)





            
            cv2.imshow(windowName2, frame2)
            ret3, frame3 = capture3.read()
            frame_id_3 += 1
            # Detecting objects
            blob_3 = cv2.dnn.blobFromImage(frame3, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_3)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width3)
                        center_y = int(detection[1] * height3)
                        w = int(detection[2] * width3)
                        h = int(detection[3] * height3)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame3, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame3, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)



            elapsed_time = time.time() - starting_time
            fps = frame_id_3 / elapsed_time
            cv2.putText(frame3, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)



            
            cv2.imshow(windowName3, frame3)
            

            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        
        cv2.destroyAllWindows()

    else:
        print("Invalid option entered. Exiting...")


main()
