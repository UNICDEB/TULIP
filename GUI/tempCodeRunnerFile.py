   # def coordinate(self, depth_data):
    #     co_ordinate = []
    #     x, y = 126, 236
    #     depth_intrin = depth_data.profile.as_video_stream_profile().intrinsics
    #     depth_value = depth_data.get_distance(x,y)
    #     d_point = rs.rs2_deproject_pixel_to_point(depth_intrin,x,y, depth_value)
    #     # depth_intrinsics = depth_frame.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    #     # depth_frame = np.loadtxt('frame_0.txt')
    
    #     # x, y, z = point[0], point[1], point[1]
    #     # co_ordinate.append(x)
    #     # co_ordinate.append(", ")
    #     # co_ordinate.append(y)
    #     # co_ordinate.append(', ')
    #     # co_ordinate.append(z)
    #     print("Coordinate - ", d_point)

    #     return(co_ordinate)
