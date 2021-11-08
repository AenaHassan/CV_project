import cv2


def main():

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


        capture1 = cv2.VideoCapture("http://10.104.6.247:8080/video")  # phone 1 camera
        capture2 = cv2.VideoCapture(0)   # laptop camera
        capture3 = cv2.VideoCapture("http://10.104.2.33:8080/video") #phone 2 camera



    
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
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
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

        capture1 = cv2.VideoCapture("http://10.104.2.33:8080/video")  # phone camera
        capture2 = cv2.VideoCapture(0)  # laptop's camera
        capture3 = cv2.VideoCapture("http://10.104.5.129:4747/video")

        if capture1.isOpened():  # check if feed exists or not for camera 1
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
            
        else:
            ret1 = False
            ret2 = False
            ret3 = False

        while ret1 and ret2 and ret3:
            ret1, frame1 = capture1.read()
            cv2.imshow(windowName1, frame1)
            ret2, frame2 = capture2.read()
            cv2.imshow(windowName2, frame2)
            ret3, frame3 = capture3.read()
            cv2.imshow(windowName3, frame3)

            if cv2.waitKey(1) == 27:
                break

        capture1.release()
        
        cv2.destroyAllWindows()

    else:
        print("Invalid option entered. Exiting...")


main()
