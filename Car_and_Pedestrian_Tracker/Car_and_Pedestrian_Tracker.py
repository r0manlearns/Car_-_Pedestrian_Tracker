import cv2
from random import randrange

vidorstream_file = "D:\Organize these\Library\GITHUB\Data-Science-Notes-n-Projects\Projects\ML-AI_Projects\skfirst_car_and_pedestrian_Tracker\carvid.mp4"
vid_or_stream = cv2.VideoCapture(vidorstream_file)
img_file = "D:\Organize these\Library\GITHUB\Data-Science-Notes-n-Projects\Projects\ML-AI_Projects\skfirst_car_and_pedestrian_Tracker\car.png"
img = cv2.imread(img_file)
car_classifier_file = 'D:\Organize these\Library\GITHUB\Data-Science-Notes-n-Projects\Projects\ML-AI_Projects\skfirst_car_and_pedestrian_Tracker\cars.xml'
car_tracker = cv2.CascadeClassifier(car_classifier_file)
pedestrian_classifier_file = 'D:\Organize these\Library\GITHUB\Data-Science-Notes-n-Projects\Projects\ML-AI_Projects\skfirst_car_and_pedestrian_Tracker\haarcascade_fullbody.xml'
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier_file)

print("img, video, or stream")

user_choice = input().lower()

if user_choice == str("img"):
    black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_tracker.detectMultiScale(black_n_white)
    pedestrians = pedestrian_tracker.detectMultiScale(black_n_white)
    print("Cars: ", cars)
    print("Pedestrians: ", pedestrians)
    
    for (x, y, w, h) in cars:
        cv2.imshow("Car Detector", img)

        cv2.rectangle(img, (x + 1, y + 1), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)      
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  
            break
        
elif user_choice == "video" or user_choice == "stream":
    while True:
        read_successful, frame = vid_or_stream.read()
        if not read_successful:
            print("Error")
            break
        
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_tracker.detectMultiScale(grayscaled_frame)
        pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
        print("Pedestrians: ", pedestrians)
        print("Cars: ", cars)
        
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x + 1, y + 1), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)
        cv2.imshow("Car Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  
            break
else:
    print("Try again and input properly, dumbass")
    quit()

print("Code Completed")