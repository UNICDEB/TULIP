import cv2
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import os
import time
import glob
from ultralytics import YOLO
from utils import *
import string

counter=0
client=Mqtt_Node(1883,'localhost')
preset = 4
exp = 80

# Depth camera
class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.cfg = self.pipeline.start(config)
        self.depth_sensor = self.cfg.get_device().first_depth_sensor()
        self.color_sensor = self.cfg.get_device().first_color_sensor()
        self.update_preset(preset)
        self.update_exposure(exp, auto=True)

    def update_preset(self, preset_value):
        self.depth_sensor.set_option(rs.option.visual_preset, preset_value)

    def update_exposure(self, exposure_value, auto=False):
        if auto:
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        else:
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            self.color_sensor.set_option(rs.option.exposure, exposure_value)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image
    
    def coordinate(self, depth_frame):
        co_ordinate = [23,45,45]
        depth_intrinsics = self.pipeline.get_active_profile().get_stream(
            rs.stream.depth).as_video_stream_profile().get_intrinsics()
        # depth_frame = np.loadtxt('frame_0.txt')
        depth_frame = np.asanyarray(depth_frame.get_data())
        x, y = 126, 236
        depth = depth_frame.get_distance(x, y)
        point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x,y], depth)
        x, y, z = point[0], point[1], point[1]
        co_ordinate.append(x)
        co_ordinate.append(", ")
        co_ordinate.append(y)
        co_ordinate.append(', ')
        co_ordinate.append(z)

        return(co_ordinate)


    def show_intrinsics(self):
        depth_intrinsics = self.pipeline.get_active_profile().get_stream(
            rs.stream.depth).as_video_stream_profile().get_intrinsics()
        return depth_intrinsics

    def camera_cordinates(self, u, v, ppx, ppy, fx, fy, depth):
        x = ((u - ppx) * depth) / (fx)
        y = ((v - ppy) * depth) / (fy)
        z = depth
        return x, y, z
    
    def release(self):
        self.pipeline.stop()
    

customtkinter.set_appearance_mode("System")  
customtkinter.set_default_color_theme("blue")

# Top Level Window
class ToplevelWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("300x200")
        self.title("Co-ordinate Input")
        self.label = customtkinter.CTkLabel(self, text="")
        self.label.pack(padx=20, pady=20)

# Tkinter API
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("TULIP: Tea Plucking System")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="TULIP", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text = "Open Camera", command=self.OpenCamera_btn)
        self.sidebar_button_1.grid(row=1, column=0, padx=10, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text = "Capture Image", command=self.CaptureImage_btn)
        self.sidebar_button_2.grid(row=2, column=0, padx=10, pady=10)
        self.appearance_mode_optionemenu1 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=['0', '1', '2', '3', '4'], command=self.change_preset_event)
        self.appearance_mode_optionemenu1.grid(row=3, column=0, padx=10, pady=(10, 10))
        self.appearance_mode_optionemenu2 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Auto", "Manual"], command=self.change_exposure_event)
        self.appearance_mode_optionemenu2.grid(row=4, column=0, padx=10, pady=(10, 10))
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Start", command=self.operation_btn)
        self.sidebar_button_3.grid(row=5, column=0, padx=10, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Mode:", anchor="w")
        self.appearance_mode_label.grid(row=15, column=0, padx=5, pady=(5, 5))
        self.appearance_mode_optionemenu3 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu3.grid(row=16, column=0, padx=5, pady=(5, 5))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="Screen Scaling:", anchor="w")
        self.scaling_label.grid(row=17, column=0, padx=5, pady=(5, 5))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=18, column=0, padx=5, pady=(5, 5))

        # create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text=" ", height=50, corner_radius = 10)
        self.entry.grid(row=3, column=1, columnspan=2,rowspan =10, padx=(20, 20), pady=(20, 20),sticky="nsew")
        
        # create textboxs
        self.textbox = customtkinter.CTkFrame(self)
        self.textbox.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
        self.heading = customtkinter.CTkLabel(font=("Times New Roman", 20) , justify="center" , master=self.textbox, text="\t\tWELCOME TO TULIP \n\n\t\tTea Harvesting Unmanned Robotic Plartform.")
        self.heading.grid(padx = 5,pady=5, sticky= "nswe")
        
        

        # create radiobutton frame
        self.radiobutton_frame = customtkinter.CTkFrame(self)
        self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.radio_var = tkinter.IntVar(value=0)
        self.label_radio_group = customtkinter.CTkLabel(master=self.radiobutton_frame, text="COORDINATE INPUT :")
        self.label_radio_group.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky="nsew")
        self.radio_button_1 = customtkinter.CTkButton(self.radiobutton_frame, text="Coordinate input", command= self.open_toplevel)
        self.radio_button_1.grid(row=1, column=2, padx=20, pady= (10,10))
        self.radio_button_2 = customtkinter.CTkButton(self.radiobutton_frame, text="Exit", command= self.quit)
        self.radio_button_2.grid(row=2, column=2, padx=20, pady= (10,10))
        self.entry1 = customtkinter.CTkEntry(self.radiobutton_frame, placeholder_text="X Point - ")
        self.entry1.grid(row=4, column=2, padx=20, pady=(10, 10), sticky="nsew")
        self.entry2 = customtkinter.CTkEntry(self.radiobutton_frame, placeholder_text="Y Point - ")
        self.entry2.grid(row=5, column=2, padx=20, pady=(10, 10), sticky="nsew")
        self.entry3 = customtkinter.CTkEntry(self.radiobutton_frame, placeholder_text="Z Point - ")
        self.entry3.grid(row=6, column=2, padx=20, pady=(10, 10), sticky="nsew")
        self.submit_button_1 = customtkinter.CTkButton(master=self.radiobutton_frame, text = "Submit", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE") , command=self.coordinate_input)
        self.submit_button_1.grid(row=8, column=2, padx=20, pady=(10, 10), sticky="nsew")

        # set default values
        self.appearance_mode_optionemenu1.set("Camera Preset")
        self.appearance_mode_optionemenu3.set("Dark")
        self.appearance_mode_optionemenu2.set("Exposure")
        self.scaling_optionemenu.set("100%")
        self.label = customtkinter.CTkLabel(master=self, text='')

        # Depth Camera
        # Initialize variables
        self.camera_opened = False
        self.camera = None
        
    def open_toplevel(self):
        self.toplevel_window = ToplevelWindow(self)
        self.entry1 = customtkinter.CTkEntry(self.toplevel_window, placeholder_text="X Point - ")
        self.entry1.pack()
        self.submit_button_1 = customtkinter.CTkButton(master=self.toplevel_window, text = "Submit", fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command = self.open_toplebel_input)
        self.submit_button_1.pack(padx=5, pady=(5, 5))
        

        #if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
        #    self.toplevel_window = ToplevelWindow(self)  # create window if its None or destroyed
        #else:
        #    self.toplevel_window.focus()  # if window exists focus it
    # open_toplevel function work 
    def open_toplebel_input(self):
        x = self.entry1.get()
        # y = self.entry2.get()
        # z = self.entry3.get()
        print("X Value - ", x)
        print(type(x))
        x_list = x.split(", ")
        x_list = [int(i) for i in x_list]
        client.publish(x_list, "coordinate")
        # try:
        #     self.entry.delete(0,tk.END)
        #     self.entry.insert(0,print("X, Y, Z coordinates : ", x,y,z))
        # except Exception as e:
        #     self.entry.delete(0,tk.END)
        #     self.entry.insert(0,e)

    def OpenCamera_btn(self):
        self.camera_opened = True
        self.camera = DepthCamera()
        self.update_camera()
    
    # For Color Image Function
    def update_camera(self):
        flag = 1
        #print(flag)
        ret, depth_frame, color_frame = self.camera.get_frame()
        if ret:
            if(flag==1):
                image = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                # if(flag==0):
                #     image = Image.fromarray(depth_frame)        
            photo = customtkinter.CTkImage(image , size=(650, 450))
            self.label.configure(image=photo)
            self.label.image = photo
            self.label.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")                  
        self.label.after(10, self.update_camera)

    def change_preset_event(self, new_preset):
        global preset
        preset = int(new_preset)
        if self.camera_opened:
            self.camera.update_preset(preset)

    def change_exposure_event(self, exposure_mode):
        global exp
        if exposure_mode == "Auto":
            if self.camera_opened:
                self.camera.update_exposure(exp, auto=True)
        else:
            exposure_value = customtkinter.CTkInputDialog(text="Enter Exposure Value:", title="Manual Exposure Setting").get_input()
            if exposure_value:
                exp = int(exposure_value)
                if self.camera_opened:
                    self.camera.update_exposure(exp, auto=False)
    # Camera Access

    def show_box(self,data):
        self.entry.delete(0, tk.END)
        self.entry.insert(0,data)

    # Image Capture
    def phto_capture(self):
        ret, depth_frame, color_frame = self.camera.get_frame()
        counter=0
        if ret:
            filename =  f"frame_{counter}.jpg"
            cv2.imwrite(filename, color_frame)
            f = open(f"frame_{counter}.txt", "a+")
            np.savetxt(f"frame_{counter}.txt",depth_frame)
            f.close()
            print("Depth frame_{}.txt image saved".format(counter))
            counter+=1
            tkinter.messagebox.showinfo("Image Saved", f"Image saved as {filename}")
            file1= open("test.txt","w")
            file1.write(str(counter))
            print(" Color frame_{}.jpg image saved".format(counter))

    # Object Detection Function(Using YoloV8 Algorithm)
    def detection(self):
        ###########################################
        # Photo Capture
        ret, depth_frame, color_frame = self.camera.get_frame()
        counter=0
        if ret:
            filename =  f"frame_{counter}.jpg"
            cv2.imwrite(filename, color_frame)
            f = open(f"frame_{counter}.txt", "a+")
            np.savetxt(f"frame_{counter}.txt",depth_frame)
            f.close()
            print("Depth frame_{}.txt image saved".format(counter))
            counter+=1
            tkinter.messagebox.showinfo("Image Saved", f"Image saved as {filename}")
            file1= open("test.txt","w")
            file1.write(str(counter))
            print(" Color frame_{}.jpg image saved".format(counter))

        ###############################################

        frame = cv2.imread("frame_0.jpg")
        # depth_frame = np.loadtxt("frame_0.txt")
        l1= []
        l2=[]
        model = YOLO('yolov8_weights/best.pt')
        results = model(frame)
        #annotated_frame = results[0].plot()
        #bounding_box = results[0]
        # Extract bounding boxes, classes, names, and confidences
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        names = results[0].names
        confidences = results[0].boxes.conf.tolist()
        print("All Confidence value - ", confidences)

        # Clear previous detections from the list
        # l1 - store all coordinate values
        # l2 - store all detected center points
        l1.clear()
        l2.clear()
        
        # Add new detections to the list if their confidence is above 50%
        for box, confidence in zip(boxes, confidences):
            if confidence >= 0.1:
                l1.append(box)
                # Center Point Calcute
                center_x = round((box[0] + box[2]) / 2)
                center_y = round((box[1] + box[3]) / 2)
                l2.append((center_x, center_y))

        if(len(l2)>0):
            self.entry.delete(0,tk.END)
            self.entry.insert(0, l2)
        else:
            self.entry.delete(0, tk.END)
            self.entry.insert(0,"No Object Found")

        l3 = self.camera.coordinate(depth_frame)    
        # Data send
        # l3=[1,2,3,4,5,6]
        client.publish(l3 , "coordinate")
        return l1, l2

    # Pixel to Conordinte Convertion
    # def coordinate():
    #     depth = depth_frame.get_distance(x, y)
    #     point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x,y], depth)
    #     x, y, z = point[0], point[1], point[1]
        
    # Camera Is Running - show in a text box -- Open Camera Button Hit    
    # def OpenCamera_btn(self):
    #     self.color_frme()
    #     self.entry.delete(0, tk.END)
    #     self.entry.insert(0,"Camera Succesfully Opened....")
        
    # Capture_Image Button Hit   
    def CaptureImage_btn(self):
        self.phto_capture()   
        self.entry.delete(0, tk.END)
        a= self.entry.insert(0,"Current Frame Succesfully Saved...............")
        #print(a)

    # Refresh Button Hit     
    def operation_btn(self):
        self.detection()
        self.entry.delete(0, tk.END)
        self.entry.insert(0,"Operation Successfully Complete")  

        
    def open_input_dialog_event(self):
        dialog1 = customtkinter.CTkInputDialog(text="Type x, y, z coordinates:", title="CoordinateValue")
        print("X, Y, Z coordinates : ", dialog1.get_input())
        print("Type - ", type(dialog1.get_input()))
        # value =dialog1.get_input()
        client.publish(dialog1.get_input(), "coordinate")

    def coordinate_input(self):
        x = self.entry1.get()
        y = self.entry2.get()
        z = self.entry3.get()
        print("X Value - ", x)
        print("Y Value - ", y)
        print("Z VAlue - ", z)
        value =[int(x), int(y), int(z)]
        print(type(value[0]))
        
        client.publish(value, "coordinate")
        try:
            self.entry.delete(0,tk.END)
            self.entry.insert(0,print("X, Y, Z coordinates : ", x,y,z))
        except Exception as e:
            self.entry.delete(0,tk.END)
            self.entry.insert(0,e)
            
    def change_appearance_mode_event(self,new_appearance_mode: str):    
        customtkinter.set_appearance_mode(new_appearance_mode)
        if(new_appearance_mode=="Auto"):
            self.entry.delete(0, tk.END)
            a= self.entry.insert(0,"Auto Button Hit......")
        elif(new_appearance_mode=="Manual"):
            manual_exp_value = customtkinter.CTkInputDialog(text="Enter Exposure value(0-200)", title="Manual Exposure Value")
            print("Exposure Value - ", manual_exp_value.get_input())
            #exp_val = manual_exp_value.get_input()
            #print(exp_val)
            #self.entry.delete(0, tk.END)
            #self.entry.insert(0, manual_exp_value.get_input())

        elif(new_appearance_mode=="0"):
            preset = 0
            self.entry.delete(0,tk.END)
            self.entry.insert(0,preset)
        elif(new_appearance_mode=="1"):
            preset = 1
            self.entry.delete(0,tk.END)
            self.entry.insert(0,preset)
        elif(new_appearance_mode=="2"):
            preset = 2
            self.entry.delete(0,tk.END)
            self.entry.insert(0, "bitton- 2")
        elif(new_appearance_mode=="3"):
            preset = 3
            self.entry.delete(0,tk.END)
            self.entry.insert(0, "button- 3")
        elif(new_appearance_mode=="4"):
            preset = 4
            self.entry.delete(0,tk.END)
            self.entry.insert(0, "button- 4")

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)


if __name__ == "__main__":

    # # Full Module (final version of code)
    # try:
    #     dc = DepthCamera()
    # except:
    #     print("Error: Depth Camera not connected")
   
    client.connect_mqtt()
    app = App()
    client.subscribe(app)
    client.start_listining() 

    #app = App()
    app.mainloop()
