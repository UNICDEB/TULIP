import pyrealsense2 as rs
import cv2
import numpy as np

# Create a pipeline object
pipeline = rs.pipeline()

# Create a config object and specify the streams of interest
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from the camera
pipeline.start(config)

# Define a callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the depth frame
        depth_frame = param[0]
        # Get the x, y, and z coordinates of the tapped pixel
        depth = depth_frame.get_distance(x, y)
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
        x, y, z = point[0], point[1], point[2]
        coordinates = [x, y, z]
        print("The x, y, z coordinates of the tapped pixel are:", coordinates)

try:
    # Wait for a frame to arrive
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    # Get the intrinsic parameters of the depth camera
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_frame = frames.get_color_frame()
    # Convert the color frame to an OpenCV image
    color_image = np.asanyarray(color_frame.get_data())
    # Create a window and set the mouse callback function
    cv2.namedWindow("Color Stream", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Color Stream", mouse_callback, [depth_frame])
    while True:
        cv2.imshow("Color Stream", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    
    cv2.destroyAllWindows()
