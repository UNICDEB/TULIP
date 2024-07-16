# import pyrealsense2 as rs
# import numpy as np
# import cv2

# global coordinate

# class RealSenseCamera:
#     def __init__(self):
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
#         self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
#         self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
#         self.pipeline.start(self.config)

#     def get_frames(self):
#         return self.pipeline.wait_for_frames()

#     def get_depth_frame(self, frames):
#         return frames.get_depth_frame()

#     def get_color_frame(self, frames):
#         return frames.get_color_frame()

#     def get_depth_intrinsics(self, depth_frame):
#         return depth_frame.profile.as_video_stream_profile().intrinsics

#     def stop(self):
#         self.pipeline.stop()

# class MouseCallback:
    
#     def __init__(self, depth_intrin, depth_frame):
#         self.depth_intrin = depth_intrin
#         self.depth_frame = depth_frame

#     def callback(self, event, x, y):
#         coordinate = []
        
#         if event == cv2.EVENT_LBUTTONDOWN:
#             depth = self.depth_frame.get_distance(x, y)
#             point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x, y], depth)
#             x, y, z = point[0], point[1], point[2]
#             print("X points - ", round(x*1000))
#             print("Y Points - ", round(y*1000))
#             print("Z points - ", round(z*1000))
            

# def main():
#     camera = RealSenseCamera()
#     try:
#         while True:
#             frames = camera.get_frames()
#             depth_frame = camera.get_depth_frame(frames)
#             depth_intrin = camera.get_depth_intrinsics(depth_frame)
#             color_frame = camera.get_color_frame(frames)
#             color_image = np.asanyarray(color_frame.get_data())
#             cv2.namedWindow("Color Stream", cv2.WINDOW_NORMAL)
#             mouse_callback = MouseCallback(depth_intrin, depth_frame)
#             cv2.setMouseCallback("Color Stream", mouse_callback.callback)
#             cv2.imshow("Color Stream", color_image)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         camera.stop()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

####################################

import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

    def get_frames(self):
        return self.pipeline.wait_for_frames()

    def get_depth_frame(self, frames):
        return frames.get_depth_frame()

    def get_color_frame(self, frames):
        return frames.get_color_frame()

    def get_depth_intrinsics(self, depth_frame):
        return depth_frame.profile.as_video_stream_profile().intrinsics

    def stop(self):
        self.pipeline.stop()

class MouseCallback:
    def __init__(self, depth_intrin, depth_frame, coordinate_list):
        self.depth_intrin = depth_intrin
        self.depth_frame = depth_frame
        self.coordinate_list = coordinate_list

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth = self.depth_frame.get_distance(x, y)
            point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x, y], depth)
            x, y, z = point[0], point[1], point[2]

            final_x = round(x*1000)
            final_y = round(y*1000)
            final_z = round(z*1000)

            print("X points - ", final_x)
            print("Y Points - ", final_y)
            print("Z points - ", final_z)
            self.coordinate_list.append(final_x)
            self.coordinate_list.append(final_y)
            self.coordinate_list.append(final_z)

def main():
    camera = RealSenseCamera()
    coordinate_list = []
    try:
        while True:
            frames = camera.get_frames()
            depth_frame = camera.get_depth_frame(frames)
            # Apply depth filter
            # Apply spatial filter (hole filling) and temporal filter
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            depth_frame = spatial.process(depth_frame)
            # temporal = rs.temporal_filter()
            # depth_frame = temporal.process(depth_frame)
            depth_intrin = camera.get_depth_intrinsics(depth_frame)
            color_frame = camera.get_color_frame(frames)
            color_image = np.asanyarray(color_frame.get_data())
            cv2.namedWindow("Color Stream", cv2.WINDOW_NORMAL)
            mouse_callback = MouseCallback(depth_intrin, depth_frame, coordinate_list)
            cv2.setMouseCallback("Color Stream", mouse_callback.callback)
            cv2.imshow("Color Stream", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Coordinate List: ", coordinate_list)

if __name__ == "__main__":
    main()