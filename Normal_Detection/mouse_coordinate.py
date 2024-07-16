import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf



def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_frame = param[0]
        print("Param - ", depth_frame)
        depth = depth_frame.get_distance(x, y)
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
        x, y, z = point[0], point[1], point[2]
        print("X points - ", round(x*1000))
        print("Y Points - ", round(y*1000))
        print("Z points - ", round(z*1000))
        # coordinates = [x, y, z]
        coordinates = [round(x*1000), round(y*1000), round(z*1000)]
        if (z>0):
            l1.append(coordinates)
        print("The x, y, z coordinates of the tapped pixel are:", coordinates)
        print("Type - ", type(coordinates))
        return l1
    
# Final Coordinate Calculations
def final_coordinate(l1):
    # Load the saved model
    loaded_model = tf.keras.models.load_model(r'/home/aeebot/Desktop/TULIP_Code/GUI/model_best.keras')
    print(loaded_model)
    z_value = [sublist[-1] for sublist in l1]
    new_data = [sublist[:-1] for sublist in l1] # Here new data is (x,y) pixel value
    new_data = np.array(new_data)
    new_data=new_data/100
    new_data=tf.convert_to_tensor(new_data, dtype=tf.float32)
    print(new_data)
    predictions = loaded_model.predict(new_data)
    predictions = predictions*100
    output = []
    for i in range(len(predictions)):
        output.append(int(predictions[i][0]))
        output.append(int(predictions[i][1]))
        output.append(z_value[i])
    print("Prediction - ",output)
    # self.ofset(predictions,10,10)
    return(output)



if __name__=="__main__":
    l1 = []
    l1.clear()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    # # Create filters
    # spatial_filter = rs.spatial_filter()
    # temporal_filter = rs.temporal_filter()
    # hole_filling_filter = rs.hole_filling_filter()
    # decimation_filter = rs.decimation_filter()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            #  # Apply filters
            # depth_frame = decimation_filter.process(depth_frame)  # Decimation filter
            # depth_frame = spatial_filter.process(depth_frame)  # Spatial filter
            # depth_frame = temporal_filter.process(depth_frame)  # Temporal filter
            # depth_frame = hole_filling_filter.process(depth_frame)  # Hole filling filter


            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            cv2.namedWindow("Color Stream", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Color Stream", mouse_callback, [depth_frame])
            cv2.imshow("Color Stream", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    print("Coordinates - ", l1)
    fnl_cood = final_coordinate(l1)
    print("Final Values - ", fnl_cood)


##################################
# import pyrealsense2 as rs
# import numpy as np
# import cv2

# # Initialize the pipeline
# pipeline = rs.pipeline()
# config = rs.config()

# # Configure the pipeline to stream depth and color
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

# # Create filters
# spatial_filter = rs.spatial_filter()
# temporal_filter = rs.temporal_filter()
# hole_filling_filter = rs.hole_filling_filter()
# decimation_filter = rs.decimation_filter()

# # Create an alignment object
# align_to_color = rs.align(rs.stream.color)

# # List to store coordinates
# coordinates_list = []

# # Mouse callback function
# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Check if the coordinates are within bounds
#         if x < color_image.shape[1] and y < color_image.shape[0]:
#             # Map the color frame coordinates to the aligned depth frame
#             depth_frame = aligned_frames.get_depth_frame()
#             if not depth_frame:
#                 print("Depth frame is not available")
#                 return

#             # Retrieve the depth value
#             depth_value = depth_frame.get_distance(x, y)

#             # Append coordinates to list
#             coordinates_list.append((x, y, depth_value))

#             print(f"Mouse clicked at ({x}, {y}), Depth: {depth_value} meters")
#         else:
#             print(f"Mouse click out of bounds: ({x}, {y})")

# try:
#     while True:
#         # Wait for a frame
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align_to_color.process(frames)
        
#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()

#         # Apply filters to depth frame
#         depth_frame = decimation_filter.process(depth_frame)  # Decimation filter
#         depth_frame = spatial_filter.process(depth_frame)  # Spatial filter
#         depth_frame = temporal_filter.process(depth_frame)  # Temporal filter
#         depth_frame = hole_filling_filter.process(depth_frame)  # Hole filling filter

#         # Convert depth frame to numpy array
#         depth_image = np.asanyarray(depth_frame.get_data())

#         # Convert color frame to numpy array
#         color_image = np.asanyarray(color_frame.get_data())

#         # Display images
#         cv2.imshow('Color Frame', color_image)
#         cv2.imshow('Depth Frame', depth_image)

#         # Set up the mouse callback
#         cv2.setMouseCallback('Color Frame', mouse_callback)

#         # Break loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     # Stop streaming
#     pipeline.stop()
#     cv2.destroyAllWindows()

#     # Print out the stored coordinates
#     print("Stored coordinates (x, y, z) in meters:")
#     for coord in coordinates_list:
#         print(coord)

#######################################
# import pyrealsense2 as rs
# import numpy as np
# import cv2

# # Initialize the pipeline
# pipeline = rs.pipeline()
# config = rs.config()

# # Configure the pipeline to stream depth and color
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

# # Create filters
# spatial_filter = rs.spatial_filter()
# temporal_filter = rs.temporal_filter()
# hole_filling_filter = rs.hole_filling_filter()
# decimation_filter = rs.decimation_filter()

# # Create an alignment object
# align_to_color = rs.align(rs.stream.color)

# # List to store coordinates
# coordinates_list = []

# # Mouse callback function
# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Check if the coordinates are within bounds
#         if x < color_image.shape[1] and y < color_image.shape[0]:
#             # Map the color frame coordinates to the aligned depth frame
#             depth_frame = aligned_frames.get_depth_frame()
#             if not depth_frame:
#                 print("Depth frame is not available")
#                 return

#             # Get depth value at the pixel location
#             depth_value = depth_frame.get_distance(x, y)

#             # Convert depth value to real-world coordinates (x, y, z)
#             # Note: Depth value is in meters
#             z = depth_value  # Depth in meters
#             x_world = (x - color_intrinsics.ppx) / color_intrinsics.fx * z
#             y_world = (y - color_intrinsics.ppy) / color_intrinsics.fy * z

#             # Append coordinates to list
#             coordinates_list.append((x_world, y_world, z))

#             print(f"Mouse clicked at ({x}, {y}), Depth: {depth_value} meters")
#             print(f"Real-world coordinates: ({x_world}, {y_world}, {z}) meters")

#         else:
#             print(f"Mouse click out of bounds: ({x}, {y})")

# try:
#     while True:
#         # Wait for a frame
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align_to_color.process(frames)
        
#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()

#         # Apply filters to depth frame
#         depth_frame = decimation_filter.process(depth_frame)  # Decimation filter
#         depth_frame = spatial_filter.process(depth_frame)  # Spatial filter
#         depth_frame = temporal_filter.process(depth_frame)  # Temporal filter
#         depth_frame = hole_filling_filter.process(depth_frame)  # Hole filling filter

#         # Convert depth frame to numpy array
#         depth_image = np.asanyarray(depth_frame.get_data())

#         # Convert color frame to numpy array
#         color_image = np.asanyarray(color_frame.get_data())

#         # Get the intrinsics of the color camera
#         color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

#         # Display images
#         cv2.imshow('Color Frame', color_image)
#         cv2.imshow('Depth Frame', depth_image)

#         # Set up the mouse callback
#         cv2.setMouseCallback('Color Frame', mouse_callback)

#         # Break loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     # Stop streaming
#     pipeline.stop()
#     cv2.destroyAllWindows()

#     # Print out the stored coordinates
#     print("Stored coordinates (x, y, z) in meters:")
#     for coord in coordinates_list:
#         print(coord)



