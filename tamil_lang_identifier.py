import math
import cv2
import numpy as np
import random as rand
from collections import deque
import sys



def plotter():

    cap = cv2.VideoCapture(0)
    # To keep track of all point where object visited
    center_points = deque()

    kernel = np.ones((15,15),np.uint8)



    while True:
        # Read and flip frame
        # Read and flip frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)



        # Blur the frame a little
        # blur_frame = cv2.GaussianBlur(frame, (7, 7),2)
        blur_frame = cv2.medianBlur(frame, 15)

        # Convert from BGR to HSV color format
        hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

        # Define lower and upper range of hsv color to detect. Blue here
        lower_blue = np.array([165, 150, 150])
        upper_blue = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        erosion = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)



        # Find all contours
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(contours) > 0:
            # Find the biggest contour
            biggest_contour = max(contours, key=cv2.contourArea)

            # Find center of contour and draw filled circle
            moments = cv2.moments(biggest_contour)
            try:
                centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                cv2.circle(frame, centre_of_contour, 5, (0, 0, 255), -1)

                # Bound the contour with circle
                ellipse = cv2.fitEllipse(biggest_contour)
                cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

                # Save the center of contour so we draw line tracking it
                center_points.appendleft(centre_of_contour)

            except Exception as e:
                print(e)


        # Draw line from center points of contour
        for i in range(1, len(center_points)):
            if math.sqrt(((center_points[i - 1][0] - center_points[i][0]) ** 2) + (
                    (center_points[i - 1][1] - center_points[i][1]) ** 2)) <= 50:
                cv2.line(frame, center_points[i - 1], center_points[i], (0, 0, 255), 4)

        cv2.imshow('original', frame)
        cv2.imshow('mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('n'):
            break

        if cv2.waitKey(1) & 0xFF == ord('w'):
            # for data
            print('writing...')
            lower_redd = np.array([0, 0, 252])
            upper_redd = np.array([0, 0, 255])
            mask = cv2.inRange(frame, lower_redd, upper_redd)
            res = cv2.bitwise_and(frame, frame, mask=mask)
            number = rand.randint(0,9999999)
            cv2.imwrite(str(number) +'.jpg',res)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit()


while True:
    plotter()
