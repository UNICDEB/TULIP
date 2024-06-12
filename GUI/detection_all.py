## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import threading
import os
import time
import glob
from ultralytics import YOLO

global pipe
global processed_frame
global stop
global point


def detection(frame,model, l1, l2, counter): 
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imwrite(f"Result_Image/res_frame{counter}.jpg", annotated_frame)
    bounding_box = results[0]
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    print("All Confidence value - ", confidences)
    
    # Add new detections to the list if their confidence is above 50%
    if(len(boxes)>len(l1)):
        l1.clear()
        for box, confidence in zip(boxes, confidences):
            if confidence >= 0.1:
                l1.append(box)
                # Center Point Calcute
                center_x = round((box[0] + box[2]) / 2)
                center_y = round((box[1] + box[3]) / 2)
                l2.append((center_x, center_y))
    
    return l1, l2


# Exp. Image Generator Function
def exp(detected_image):
    # Create a list of exposure values
    exposures = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # Create a list of images with different exposures
    exposed_images = []
    for exposure in exposures:
        exposed_image = cv2.convertScaleAbs(detected_image, alpha=exposure, beta=0)
        exposed_images.append(exposed_image)
    # Save the images
    for i, exposed_image in enumerate(exposed_images):
        cv2.imwrite("Exposure_Image/exp_image_{}.jpg".format(i), exposed_image)


#     # Define a callback function for mouse events
# def mouse_callback(event, x, y, flags, param):
#      if event == cv2.EVENT_LBUTTONDOWN:
#         # Get the depth frame
#         global point
#         point=(x,y)
#         print(point)
        # return point
# 
# Create a window and set the mouse callback function
cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)



stop=False
color_map=rs.colorizer()
dec=rs.decimation_filter()
dec.set_option(rs.option.filter_magnitude,2)
depth2disparity=rs.disparity_transform()
disparity2depth=rs.disparity_transform(False)
spat=rs.spatial_filter()
spat.set_option(rs.option.holes_fill,5)
temp=rs.temporal_filter()
align_to=rs.align(rs.stream.color)


def post_processing_thread(lock):
    global pipe
    while(not stop):
        
        data=pipe.poll_for_frames()
        if(data):
            data=align_to.process(data)
            # print("ok")
            # data=data.get_depth_frame()
            data.as_frameset()
            # print(data.get_height())
            lock.acquire()
            data=depth2disparity.process(data)
            data=spat.process(data)
            data=temp.process(data)
            data=disparity2depth.process(data)
            processed_frame.enqueue(data)
            lock.release()

if __name__=="__main__":
    point=(400,200)
    pipe=rs.pipeline()
    cfg=rs.config()
    lock=threading.Lock()
    
    found=True
    
    cfg.enable_stream(rs.stream.depth, 848,480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
    
    profile=pipe.start(cfg)
    sensor=profile.get_device().first_depth_sensor()
    sensor.set_option(rs.option.visual_preset,4)
    
    stream=profile.get_stream(rs.stream.depth).as_video_stream_profile()
    
    processed_frame=rs.frame_queue()
    threading.Thread(target=post_processing_thread,args=(lock,)).start()
    while(True):
        # cv2.setMouseCallback("Color Stream", mouse_callback)
        # print("Inside main")
        # cv2.setMouseCallback("Frame",mouse_callback)
        current_frameset=processed_frame.poll_for_frame().as_frameset()
        if(current_frameset.is_frameset()):
            depth=current_frameset.get_depth_frame()
            color=current_frameset.get_color_frame()
            #get intrinsics
            depth_intrin = depth.profile.as_video_stream_profile().intrinsics
            #get depth value form point
            depth_value = depth.get_distance(point[0],point[1])
            
            # get global coordinates
            # Convert pixel coordinates to 3D coordinates
            # while(found):
            #     for i in range(630,650):
            #         for j in range (350,370):
            #             depth_value = depth.get_distance(i,j)
            #             d_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [i, j], depth_value)
            #             x, y, z = round(d_point[0],3),round( d_point[1],3),round( d_point[2],3)
            #             if(x==0 and y==0):
            #                 print(i)
            #                 print(j)
            #                 found=False
            #                 middle_point=[i,j]
                            
            d_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [point[0], point[1]], depth_value)
            x, y, z = round(d_point[0],3),round( d_point[1],3),round( d_point[2],3)
            # print(x,y,z)
            color_image = np.asanyarray(color.get_data())
            # cv2.circle(color_image,(middle_point[0],middle_point[1]),4,(0,0,255),2)
            # cv2.putText(color_image,"{} , {} ,{} m".format(x,y,z),(point[0],point[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
            cv2.imshow("Frame",color_image)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            captured_image=color_image
            depth_data=depth
            cv2.imwrite("Captured_image.jpg", color_image)
            stop=True
            pipe.stop()
            cv2.destroyAllWindows()
            break
        
    counter=0
     # Call Exp Function
    exp(captured_image)
    l1= []
    l2=[]
    depth_calcu_values=[]
    # Load the Best Model
    model = YOLO('weights//best.pt')
    path = "Exposure_Image"
    # List the files in the folder.
    files = os.listdir(path)
    # Read all images in the folder.
    images = []
    # Detect  Object from each image and save it to list
    for file in files:
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        counter=counter+1
        detection(img,model,l1, l2, counter)
    
    # Draw  Rectangle on Images
    for i in range(len(l1)):
        start_point = (round(l1[i][0]), round(l1[i][1]))
        end_point = (round(l1[i][2]), round(l1[i][3]))
        detected_image=cv2.rectangle(captured_image, start_point, end_point, (0,0,255), 3)
    
    middle_values=int(len(l2))
    print(middle_values)
    # for i in range(middle_values):
              
    print("Final Detected Points - ", l1)
    print("Length of Updated Values - ", len(l1))
    # end_time = time.time()  
    # print(f"Total work time - {(end_time-start_time)}")
    print("Center - ", l2)
    for i in range(middle_values):
        depth_value = depth_data.get_distance(l2[i][0],l2[i][1])
        d_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [l2[i][0],l2[i][1]], depth_value)
        depth_calcu_values.append(d_point)
    
    print(depth_calcu_values)
        
    
    # cv2.line(detected_image, (640,0), (640, 720), (255, 0, 0), 2)
    # cv2.line(detected_image, (0,360), (1280, 360), (255, 0, 0), 2)
    cv2.imwrite('Result_Image/detected_image02.jpg', detected_image)
    cv2.imshow("YOLOv8 Inference", detected_image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
    
    
    
    