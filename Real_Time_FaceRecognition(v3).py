import face_recognition as fr
import numpy as np
import pandas as pd
import cv2
import os

video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_name = []

directory = (r"E:\University\Spring\ArtificialIntelligenceLab\Project\Real_Time _Image_Detection_&_Recognition\trainImage")

for trainImage_name in os.listdir(directory):
    for trainImage_filename in os.listdir(f"{directory}/{trainImage_name}"):
        path = os.path.join(directory, trainImage_name, trainImage_filename)
        temp_img = fr.load_image_file(f"{path}")
        temp_img = fr.face_encodings(temp_img)
          
        if len(temp_img) > 0:
            known_face_encodings.append(temp_img[0])
        else:
            print("No Face Found")
            continue
        known_face_name.append(trainImage_name)

face_locations = []
face_encodings = []
face_names = []
sheet = []
process_this_frame = True

i = 0

while True:
    ret, frame = video_capture.read()
    resized_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rgb_resized_frame = resized_frame[:,:,::-1]
    
    if process_this_frame:
        face_locations = fr.face_locations(rgb_resized_frame)
        face_encodings = fr.face_encodings(rgb_resized_frame, face_locations)
        name_list = []
        face_names = []
        
        for face_encoding in face_encodings:
            matches = fr.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_name[best_match_index]
                face_names.append(name)
                
                if name not in sheet:
                    sheet.append(name)       
            else:
                name = "Unknown"
                face_names.append(name)
    i+=1
    
    if i==5:
        current_name = name
        print(current_name)
        
    if len(face_names)==0:
        i=0
    
    process_this_frame = not process_this_frame
    
    for (top,right,bottom,left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        
        cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 3)
        cv2.rectangle(frame, (left,bottom-35), (right,bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6,bottom-6), font, 1.0, (255,255,255), 2)
        
    cv2.imshow("Face Recognizing", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
   
video_capture.release()
cv2.destroyAllWindows()

output = pd.DataFrame({"Name": sheet})
output.to_csv("Attendance Sheet.csv", index=False)
print("Your submission was successfully saved")

        
    

            

