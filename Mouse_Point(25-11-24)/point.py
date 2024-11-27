import pyrealsense2 as rs
import numpy as np
import cv2
import threading

# Global variables
global pipe
global processed_frame
global stop
global point

# Define a callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        print("Selected Point:", point)

# Create a window and set the mouse callback function
cv2.namedWindow("Frame", cv2.WINDOW_AUTOSIZE)

# Initialize variables
stop = False
color_map = rs.colorizer()
dec = rs.decimation_filter()
dec.set_option(rs.option.filter_magnitude, 2)
depth2disparity = rs.disparity_transform()
disparity2depth = rs.disparity_transform(False)
spat = rs.spatial_filter()
spat.set_option(rs.option.holes_fill, 5)
temp = rs.temporal_filter()
align_to = rs.align(rs.stream.color)

# Post-processing thread function
def post_processing_thread(lock):
    global pipe
    while not stop:
        data = pipe.poll_for_frames()
        if data:
            data = align_to.process(data)
            lock.acquire()
            data = depth2disparity.process(data)
            data = spat.process(data)
            data = temp.process(data)
            data = disparity2depth.process(data)
            processed_frame.enqueue(data)
            lock.release()

if __name__ == "__main__":
   
    points_list = []
    final_list = [0]
    count = 0

    print("s- Save the RGB Frame")
    print("a- Add the Points")
    print("c- Clear all the Selected Points")
    print("t-Transfer the data From One to Another Machine")
    print("q- Close the Window & Stop the Program")

    point = (400, 200)
    pipe = rs.pipeline()
    cfg = rs.config()
    lock = threading.Lock()

    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

    profile = pipe.start(cfg)
    sensor = profile.get_device().first_depth_sensor()
    sensor.set_option(rs.option.visual_preset, 4)

    processed_frame = rs.frame_queue()
    threading.Thread(target=post_processing_thread, args=(lock,)).start()

    while True:
        cv2.setMouseCallback("Frame", mouse_callback)
        current_frameset = processed_frame.poll_for_frame().as_frameset()
        if current_frameset.is_frameset():
            depth = current_frameset.get_depth_frame()
            color = current_frameset.get_color_frame()
            depth_intrin = depth.profile.as_video_stream_profile().intrinsics
            color_image = np.asanyarray(color.get_data())

            for p in points_list:
                cv2.circle(color_image, p, 8, (68, 247, 11), -1)

            depth_value = depth.get_distance(point[0], point[1])
            d_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [point[0], point[1]], depth_value)
            x, y, z = round(d_point[0]*1000, 2), round(d_point[1] * 1000, 2), round(d_point[2] * 1000, 2)
            cv2.circle(color_image, (point[0], point[1]), 8, (0, 0, 255), -1)
            cv2.putText(color_image, "{} , {} ,{} m".format(x, y, z), (point[0], point[1] - 20),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

            cv2.imshow("Frame", color_image)

            k = cv2.waitKey(1)

            if k == ord('s'):
                cv2.imwrite("frame.jpg", color_image)

            if k == ord('a'):
                count += 1
                final_list[0]= count
                final_list.extend((x,y,z))
                points_list.append((point[0], point[1]))
                

            if k == ord("t"):
                print("t press from keyboard")

            if k == ord('c'):
                points_list = []
                final_list=[0]
                count = 0
                print("Cleared all points and reset the counter.")

            if k == ord('q'):
                stop = True
                pipe.stop()
                break

    print("Final Points -", final_list)

