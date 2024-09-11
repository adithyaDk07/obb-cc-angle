import sys
import os
import cv2
import numpy as np
import socket
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter.filedialog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import json
import torch
import math

SETTINGS_FILE = "settings.json"

class ObjectDetectionSoftware:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection Software")
        ctk.set_appearance_mode("dark")
        self.root.geometry("1680x1080")

        self.selected_model = "detectron2"
        self.custom_model_path = None
        self.predictor = None
        self.yolo_model = None
        self.obb_model = None

        self.setup_models()
        self.workspace_width_cm = 40
        self.workspace_height_cm = 23
        self.robot_socket = None
        self.capture = None
        self.captured_image = None
        self.captured_image_path = None
        self.selected_camera = 0
        self.robot_ip = ""
        self.robot_port = ""
        self.obb_results = None
        self.load_settings()
        self.create_menu_bar()
        self.create_main_layout()
        self.create_control_panel()
        self.update_ui_with_settings()
        self.output_display = []

        self.cap = cv2.VideoCapture(self.selected_camera)
        self.update_webcam_feed()

    def create_menu_bar(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        # File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        # Camera menu
        self.camera_menu = tk.Menu(self.menubar, tearoff=0)
        for i in range(5):
            self.camera_menu.add_command(label=f"Camera {i}", command=lambda x=i: self.select_camera(x))
        self.menubar.add_cascade(label="Camera", menu=self.camera_menu)

        # Model menu
        self.model_menu = tk.Menu(self.menubar, tearoff=0)

        # Create SEGMENTATION submenu
        self.segmentation_menu = tk.Menu(self.model_menu, tearoff=0)
        self.segmentation_menu.add_command(label="Detectron2", command=lambda: self.select_model("detectron2"))
        self.segmentation_menu.add_command(label="YOLO-Seg", command=lambda: self.select_model("yolov8-seg"))
        self.segmentation_menu.add_command(label="Custom SEG", command=self.load_custom_model)

        # Create OBB submenu
        self.obb_menu = tk.Menu(self.model_menu, tearoff=0)
        self.obb_menu.add_command(label="Load OBB Model", command=self.load_obb_model)

        # Add submenus to the main model menu
        self.model_menu.add_cascade(label="SEGMENTATION", menu=self.segmentation_menu)
        self.model_menu.add_cascade(label="OBB", menu=self.obb_menu)

        # Add the main model menu to the menubar
        self.menubar.add_cascade(label="Model", menu=self.model_menu)

        # Limits menu
        self.limits_menu = tk.Menu(self.menubar, tearoff=0)
        self.limits_menu.add_command(label="Set Limits", command=self.open_limits_dialog)
        self.menubar.add_cascade(label="Limits", menu=self.limits_menu)
    def setup_models(self):
        self.setup_detectron2()
        self.setup_yolov8()



    def open_limits_dialog(self):
        limits_window = ctk.CTkToplevel(self.root)
        limits_window.title("Set Limits")

        ctk.CTkLabel(limits_window, text="Width Limit (cm)").grid(row=0, column=0, padx=10, pady=10)
        width_limit_entry = ctk.CTkEntry(limits_window)
        width_limit_entry.grid(row=0, column=1, padx=10, pady=10)
        width_limit_entry.insert(0, str(self.width_limit))

        ctk.CTkLabel(limits_window, text="Height Limit (cm)").grid(row=1, column=0, padx=10, pady=10)
        height_limit_entry = ctk.CTkEntry(limits_window)
        height_limit_entry.grid(row=1, column=1, padx=10, pady=10)
        height_limit_entry.insert(0, str(self.height_limit))

        save_button = ctk.CTkButton(limits_window, text="Save", command=lambda: self.save_limits(width_limit_entry.get(), height_limit_entry.get(), limits_window))
        save_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
    def load_obb_model(self):
        model_file = tk.filedialog.askopenfilename(
            title="Select OBB Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if model_file:
            try:
                self.obb_model = YOLO(model_file)
                self.selected_model = "obb"
                self.terminal_frame.insert(tk.END, f"OBB model loaded: {model_file}\n")
            except Exception as e:
                self.terminal_frame.insert(tk.END, f"Error loading OBB model: {str(e)}\n")

    def load_limits_from_settings(self):
        default_limits = {
            'width_limit': 10.0,  # Default values
            'height_limit': 15.0  # Default values
        }

        if not os.path.exists(SETTINGS_FILE):
            # Create default settings file if it does not exist
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(default_limits, f, indent=4)

        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                # Ensure that all required keys are present
                for key in default_limits:
                    if key not in settings:
                        settings[key] = default_limits[key]

                return settings
        except Exception as e:
            print(f"Error loading limits: {str(e)}")
            # Return default limits in case of an error
            return default_limits

    def save_limits(self, width, height, dialog_window):
        try:
            self.width_limit = float(width)
            self.height_limit = float(height)
            self.save_settings()
            dialog_window.destroy()
            self.update_ui_with_settings()
            self.terminal_frame.insert(tk.END,"Limits saved successfully.\n")
        except ValueError:
            self.terminal_frame.insert(tk.END,"Invalid input. Please enter numeric values for limits.\n")
        except Exception as e:
            self.terminal_frame.insert(tk.END,f"Error saving limits: {str(e)}\n")



# Function to open a directory dialog to choose a model
    def open_model_directory(self):
    # Open file dialog to select either .pt or .h5 files
        model_file = tk.filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", ".pt *.h5"), ("All Files", ".*")]
        )
        if model_file:
            self.terminal_frame.insert(tk.END,f"Custom model selected: {model_file}")




    def select_model(self, model_name):
        self.selected_model = model_name
        self.terminal_frame.insert(tk.END,f"Selected model: {model_name}")




    def load_custom_model(self):
        model_file = tk.filedialog.askopenfilename(
            title="Select Custom Model File",
            filetypes=[("Model Files", "*.pt *.pth"), ("All Files", "*.*")]
        )
        if model_file:
            self.custom_model_path = model_file
            self.selected_model = "custom"
            self.terminal_frame.insert(tk.END,f"Custom model loaded: {model_file}")




    def create_main_layout(self):
        logo_image = Image.open(r"/home/adithyadk/Desktop/checkerBoard/primary.jpg")  # Replace with the path to your logo file
        logo_image = logo_image.resize((80, 80), Image.LANCZOS)  # Resize to 80x80
        self.logo_photo = ImageTk.PhotoImage(logo_image)

        self.logo_label = ctk.CTkLabel(self.root, image=self.logo_photo, text="")  # Set text="" to remove text under image
        self.logo_label.place(relx=0.01, rely=0.02)  # Place it in the top-left corner

        # Add AMS-INDIA text next to the logo
        self.ams_label = ctk.CTkLabel(self.root, text="AMS-INDIA", font=("montserrat", 24, "bold"))
        self.ams_label.place(relx=0.06, rely=0.028)  # Adjust relx to position it next to the logo

        # Add Object Detection System text under AMS-INDIA
        self.ods_label = ctk.CTkLabel(self.root, text="Object Detection Software", font=("montserrat", 16, "bold"))
        self.ods_label.place(relx=0.06, rely=0.060)  # Adjust relx and rely to position it directly under AMS-INDIA

        self.webcam_frame = ctk.CTkLabel(self.root, text="", corner_radius=0, bg_color="#EAEDF0")
        self.webcam_frame.place(relx=0.04, rely=0.12, relwidth=0.35, relheight=0.6)

        self.saved_image = ctk.CTkLabel(self.root, text="", corner_radius=0, bg_color="#EAEDF0")
        self.saved_image.place(relx=0.405, rely=0.12, relwidth=0.35, relheight=0.6)

        # Reduce output frame height and place the terminal below it
        self.output_frame = ctk.CTkLabel(self.root, text="OUTPUT", font=("montserrat", 14), text_color='black', corner_radius=0, bg_color="#C0C0C0")
        self.output_frame.place(relx=0.77, rely=0.12, relwidth=0.2, relheight=0.45)  # Reduced height


        
        self.bin_selection_label = ctk.CTkLabel(self.root, text="Bin Selection", font=("Arial", 16))
        self.bin_selection_label.place(relx=0.60, rely=0.80, anchor="center")

        bin_numbers = [str(i) for i in range(1, 10)]
        self.selected_bin = tk.StringVar(value=bin_numbers[0]) 

        self.bin_option_menu = ctk.CTkOptionMenu(self.root, variable=self.selected_bin, values=bin_numbers)
        self.bin_option_menu.place(relx=0.60, rely=0.82, relwidth=0.08, relheight=0.04)  

        self.terminal_frame = ctk.CTkTextbox(self.root, font=("Arial", 14), text_color='white', corner_radius=0, scrollbar_button_hover_color="#8f8b8b", bg_color="#8f8b8b")
        self.terminal_frame.place(relx=0.77, rely=0.58, relwidth=0.2, relheight=0.15)

    def create_control_panel(self):
        # Buttons: Segregation, Inspection, Capture
        self.segregation_button = ctk.CTkButton(self.root, text="Segregation", command=self.segregation)
        self.segregation_button.place(relx=0.765, rely=0.75, relwidth=0.1, relheight=0.06)

        self.inspection_button = ctk.CTkButton(self.root, text="Inspection", command=self.inspect)
        self.inspection_button.place(relx=0.878, rely=0.75, relwidth=0.1, relheight=0.06)

        self.capture_button = ctk.CTkButton(self.root, text="Capture", command=self.capture_and_process_image)
        self.capture_button.place(relx=0.77, rely=0.83, relwidth=0.2, relheight=0.06)

        # Robot Configuration inputs
        self.robot_config_label = ctk.CTkLabel(self.root, text="Robot Configuration")
        self.robot_config_label.place(relx=0.018, rely=0.75)

        self.toggle_var = ctk.BooleanVar(value=False)
        self.toggle_switch = ctk.CTkSwitch(self.root, text="", variable=self.toggle_var, command=self.toggle_inputs)
        self.toggle_switch.place(relx=0.085, rely=0.75)

        self.inputs_frame = ctk.CTkFrame(self.root)
        self.inputs_frame.place(relx=0.05, rely=0.78, relwidth=0.45, relheight=0.15)

        self.create_input_rows()

        # Add labels to display current limits
        #self.width_limit_label = ctk.CTkLabel(self.inputs_frame, text=f"Width Limit: {self.width_limit:.2f} cm")
        #self.width_limit_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        #self.height_limit_label = ctk.CTkLabel(self.inputs_frame, text=f"Height Limit: {self.height_limit:.2f} cm")
        #self.height_limit_label.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    def update_terminal(self, text):
        self.terminal_frame.insert(tk.END, f"{text}\n")
        self.terminal_frame.see(tk.END)  # Scroll to the end for the latest log


    def create_input_rows(self):
        # Workspace Dimension row
        self.label3 =ctk.CTkLabel(self.inputs_frame, text="")
        self.label3.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.dummy_entry=ctk.CTkLabel(self.inputs_frame,text="Width")
        self.dummy_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.dummy_entry2=ctk.CTkLabel(self.inputs_frame,text="Height")
        self.dummy_entry2.grid(row=0,column=2,padx=5,pady=5 ,sticky="ew")



        self.label1 = ctk.CTkLabel(self.inputs_frame, text="Workspace Dimensions (in CM)")
        self.label1.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.width_entry = ctk.CTkEntry(self.inputs_frame, placeholder_text="Width")
        self.width_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.height_entry = ctk.CTkEntry(self.inputs_frame, placeholder_text="Height")
        self.height_entry.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        self.set_dimension_button = ctk.CTkButton(self.inputs_frame, text="Set Dimension", command=self.set_dimension)
        self.set_dimension_button.grid(row=1, column=3, padx=5, pady=5, sticky="ew")


        # Workspace Dimension row
        self.label3 =ctk.CTkLabel(self.inputs_frame, text="")
        self.label3.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.dummy_entry=ctk.CTkLabel(self.inputs_frame,text="IP Address")
        self.dummy_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.dummy_entry2=ctk.CTkLabel(self.inputs_frame,text="Port")
        self.dummy_entry2.grid(row=2,column=2,padx=5,pady=5 ,sticky="ew")

        # Communication row for IP and Port
        self.label2 = ctk.CTkLabel(self.inputs_frame, text="Communication")
        self.label2.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        self.ip_entry = ctk.CTkEntry(self.inputs_frame, placeholder_text="IP Address")
        self.ip_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        self.port_entry = ctk.CTkEntry(self.inputs_frame, placeholder_text="Port")
        self.port_entry.grid(row=3, column=2, padx=5, pady=5, sticky="ew")

        self.connect_button = ctk.CTkButton(self.inputs_frame, text="Connect", command=self.connect_to_robot)
        self.connect_button.grid(row=3, column=3, padx=5, pady=5, sticky="ew")

        
        for i in range(4):
            self.inputs_frame.grid_columnconfigure(i, weight=1)


    def load_settings(self):
        default_settings = {
            'workspace_width_cm': 40.0,
            'workspace_height_cm': 23.0,
            'robot_ip': '',
            'robot_port': 0,
            'width_limit': 10.0,
            'height_limit': 15.0
        }

        if not os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(default_settings, f, indent=4)
        
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                self.workspace_width_cm = settings.get('workspace_width_cm', default_settings['workspace_width_cm'])
                self.workspace_height_cm = settings.get('workspace_height_cm', default_settings['workspace_height_cm'])
                self.robot_ip = settings.get('robot_ip', default_settings['robot_ip'])
                self.robot_port = settings.get('robot_port', default_settings['robot_port'])
                self.width_limit = settings.get('width_limit', default_settings['width_limit'])
                self.height_limit = settings.get('height_limit', default_settings['height_limit'])
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            self.workspace_width_cm = default_settings['workspace_width_cm']
            self.workspace_height_cm = default_settings['workspace_height_cm']
            self.robot_ip = default_settings['robot_ip']
            self.robot_port = default_settings['robot_port']
            self.width_limit = default_settings['width_limit']
            self.height_limit = default_settings['height_limit']

    def update_ui_with_settings(self):
        # Update workspace dimensions
        self.width_entry.delete(0, tk.END)
        self.width_entry.insert(0, str(self.workspace_width_cm))
        self.height_entry.delete(0, tk.END)
        self.height_entry.insert(0, str(self.workspace_height_cm))

        # Update robot configuration
        self.ip_entry.delete(0, tk.END)
        self.ip_entry.insert(0, self.robot_ip)
        self.port_entry.delete(0, tk.END)
        self.port_entry.insert(0, str(self.robot_port))

        # Update limits (if you have labels or entries for them)
        if hasattr(self, 'width_limit_label'):
            self.width_limit_label.configure(text=f"Width Limit: {self.width_limit:.2f} cm")
        if hasattr(self, 'height_limit_label'):
            self.height_limit_label.configure(text=f"Height Limit: {self.height_limit:.2f} cm")

    def update_entry_fields(self):
        self.width_entry.delete(0, tk.END)
        self.width_entry.insert(0, str(self.workspace_width_cm))
        
        self.height_entry.delete(0, tk.END)
        self.height_entry.insert(0, str(self.workspace_height_cm))
        
        self.ip_entry.delete(0, tk.END)
        self.ip_entry.insert(0, self.robot_ip)
        
        self.port_entry.delete(0, tk.END)
        self.port_entry.insert(0, str(self.robot_port))
    

    def toggle_inputs(self):
        if self.toggle_var.get():  # If the toggle is enabled
            # Make entry fields editable
            self.width_entry.config(state='normal')
            self.height_entry.config(state='normal')
            self.ip_entry.config(state='normal')
            self.port_entry.config(state='normal')
            self.inputs_frame.place(relx=0.02, rely=0.760, relwidth=0.45, relheight=0.15)

            # Update entry fields to reflect current variable values
            self.update_entry_fields()

            # Save variables if toggle is enabled
            try:
                self.workspace_width_cm = float(self.width_entry.get())
                self.workspace_height_cm = float(self.height_entry.get())
                self.robot_ip = self.ip_entry.get()
                self.robot_port = int(self.port_entry.get())

                # Save settings to file
                self.save_settings()
            except ValueError:
                self.terminal_frame.insert(tk.END,"Invalid input values. Please enter valid numbers.\n")
        else:  # If the toggle is disabled
            # Make entry fields read-only
            self.width_entry.config(state='readonly')
            self.height_entry.config(state='readonly')
            self.ip_entry.config(state='readonly')
            self.port_entry.config(state='readonly')
            self.inputs_frame.place_forget()

            # Update entry fields with current variable values, but do not modify variables
            self.update_entry_fields()

    def set_dimension(self):
        try:
            if self.toggle_var.get():  # Check if the toggle is enabled
                self.workspace_width_cm = float(self.width_entry.get())
                self.workspace_height_cm = float(self.height_entry.get())
                self.terminal_frame.insert(tk.END, f"Workspace dimensions set to: {self.workspace_width_cm}cm x {self.workspace_height_cm}cm \n")
                self.save_settings()  # Save settings only if the toggle is enabled
            else:
                self.terminal_frame.insert(tk.END, "Toggle is disabled. Dimensions not saved. \n")
        except ValueError:
            self.terminal_frame.insert(tk.END, "Invalid input. Please enter numeric values for width and height.\n")

    def save_settings(self):
        try:
            # Load current settings from file
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    current_settings = json.load(f)
            else:
                current_settings = {}

            # Update workspace dimensions if toggle is enabled
            if self.toggle_var.get():
                current_settings['workspace_width_cm'] = self.workspace_width_cm
                current_settings['workspace_height_cm'] = self.workspace_height_cm

            # Update robot communication settings if changed
            current_settings['robot_ip'] = self.robot_ip
            current_settings['robot_port'] = self.robot_port
            
            # Save width and height limits
            current_settings['width_limit'] = self.width_limit
            current_settings['height_limit'] = self.height_limit

            # Save updated settings back to file
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(current_settings, f, indent=4)

            self.terminal_frame.insert(tk.END,"Settings updated and saved successfully.\n")
        except Exception as e:
            self.terminal_frame.insert(tk.END,f"Error saving settings: {str(e)}\n")
            print("Exception occurred:", str(e))

    def select_camera(self, camera_index):
        self.selected_camera = camera_index
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.selected_camera)
        if not self.cap.isOpened():
            self.terminal_frame.insert(tk.END,f"Error: Could not open camera {self.selected_camera}\n")
        else:
            self.terminal_frame.insert(tk.END,f"Successfully opened camera {self.selected_camera}\n")

    def setup_detectron2(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)

    def setup_yolov8(self):
        try:
            self.yolo_model = YOLO('yolov8n-seg.pt')
            self.terminal_frame.insert(tk.END, "YOLOv8 model initialized successfully\n")
        except Exception as e:
            print( f"Error initializing YOLOv8 model: {str(e)}\n")
    def update_webcam_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.webcam_frame.configure(image=photo)
            self.webcam_frame.image = photo
        self.root.after(10, self.update_webcam_feed)

    def display_captured_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        self.saved_image.configure(image=photo)
        self.saved_image.image = photo

    def show_set_dimensions_dialog(self):
        dialog = ctk.CTkInputDialog(text="Enter workspace dimensions (width,height) in cm:", title="Set Workspace Dimensions")
        dimensions = dialog.get_input()
        if dimensions:
            try:
                width, height = map(float, dimensions.split(','))
                self.workspace_width_cm = width
                self.workspace_height_cm = height
                self.terminal_frame.insert(tk.END,f"Workspace dimensions set to: {width}cm x {height}cm\n")
            except ValueError:
                self.terminal_frame.insert(tk.END,"Invalid input. Please enter two numbers separated by a comma.\n")

    def show_robot_config_dialog(self):
        dialog = ctk.CTkInputDialog(text="Enter robot IP and port (IP,port):", title="Configure Robot")
        config = dialog.get_input()
        if config:
            try:
                ip, port = config.split(',')
                self.robot_ip = ip.strip()
                self.robot_port = int(port.strip())
                self.terminal_frame.insert(tk.END,f"Robot configuration set to: IP={self.robot_ip}, Port={self.robot_port}\n")
                self.save_settings()  # Save settings after updating
            except ValueError:
                self.terminal_frame.insert(tk.END,"Invalid input. Please enter IP and port separated by a comma.\n")

    def connect_to_robot(self):
        try:
            self.robot_ip = self.ip_entry.get()
            self.robot_port = int(self.port_entry.get())

            # Create a socket object if it doesn't already exist
            if not self.robot_socket:
                self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect to the robot
            self.robot_socket.connect((self.robot_ip, self.robot_port))
            
            # Save settings if connection is successful
            self.save_settings()
            self.terminal_frame.insert(tk.END,f"Connected to robot at IP={self.robot_ip}, Port={self.robot_port}\n")

        except ValueError:
            self.terminal_frame.insert(tk.END,"Invalid port number. Please enter a valid integer.\n")
        except socket.error as e:
            self.terminal_frame.insert(tk.END,f"Connection failed: {e}\n")
        except Exception as e:
           self.terminal_frame.insert(tk.END,f"An unexpected error occurred: {e}\n")

    def send_to_robot(self, object_data):
        if not self.robot_ip or not self.robot_port:
            # self.output_display.append("Robot IP or port not set. Please configure robot connection.")
            return
        try:
            port = int(self.robot_port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  
                s.connect((self.robot_ip, port))
                message = f"MOVE PartNo:{object_data['part_no']}, Xc:{object_data['x']:.2f}, Yc:{object_data['y']:.2f}, Angle:{object_data['angle']:.2f}"
                s.sendall(message.encode())
                self.output_display.append(f"Sent to robot: {message}")
        except ValueError:
           self.terminal_frame.insert(tk.END,"Invalid port number. Enter a valid integer.\n")
        except Exception as e:
           self.terminal_frame.insert(tk.END,f"Failed to send data to robot: {str(e)}\n")
         
    def send_inspection_data_to_robot(self, inspection_data):
        if not self.robot_ip or not self.robot_port:
            # self.output_display.append("Robot IP or port not set. Please configure robot connection.")
            return
        try:
            port = int(self.robot_port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  
                s.connect((self.robot_ip, port))
                message = f"Inspection Data: PartNo:{inspection_data['part no']} Width: {inspection_data['width']:.2f} cm, Height: {inspection_data['height']:.2f} cm, Result{inspection_data['result']}"
                s.sendall(message.encode())
                self.output_display.append(f"Sent to robot: {message}")
        except ValueError:
            self.terminal_frame.insert(tk.END,"Invalid port number. Enter a valid integer.\n")
        except Exception as e:
            self.terminal_frame.insert(tk.END,f"Failed to send data to robot: {str(e)}\n")

    def add_output_row(self, output_str):
        # Method to add output to the Tkinter output frame
        row = tk.Label(self.output_frame, text=output_str, bg="gray", fg="white", font=("Arial", 10))
        row.pack(fill="x", padx=10, pady=2)
    def add_terminal_row(self, output_str):
        self.terminal_frame.configure(text=self.terminal_frame.cget("text") + output_str + "\n")
    def clear_output(self):
        """Clears all the content from the output frame."""
        for widget in self.output_frame.winfo_children():
               widget.destroy()

    def capture_and_process_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_image = frame
            program_directory = os.path.dirname(os.path.abspath(__file__))
            
            # Set filename based on the selected model
            if self.selected_model == "obb":
                filename = "NICHAYAM.png"
            elif self.selected_model == "custom":
                filename = "NICHAYAM.png"
            else:
                filename = "VETRI.png"
        
            save_path = os.path.join(program_directory, filename)
            
            # Check if the directory exists
            if not os.path.exists(program_directory):
                os.makedirs(program_directory)
            
            if self.selected_model == "obb":
                processed_image = self._process_yolo_obb(self.captured_image)
            elif self.selected_model == "custom":
                processed_image = self._process_yolo_obb(self.captured_image)
            else:
                masks = self.process_image(self.captured_image)
                if len(masks) == 0:
                    self.add_output_row("No object detected.")
                    return
                
                mask = masks[0]
                mask_image = (mask * 255).astype('uint8')
                color_mask = cv2.applyColorMap(mask_image, cv2.COLORMAP_JET)
                processed_image = cv2.addWeighted(self.captured_image, 0.7, color_mask, 0.3, 0)
            
            # Attempt to save the image and check the result
            try:
                success = cv2.imwrite(save_path, processed_image)
                if success:
                    self.captured_image_path = save_path  # Save path of the processed image
                    self.display_captured_image(processed_image)
                    self.add_output_row(f"Image successfully saved as {filename}")
                else:
                    self.add_output_row(f"Failed to save the image as {filename}. cv2.imwrite returned False.")
            except Exception as e:
                self.add_output_row(f"Error occurred while saving the image: {str(e)}")
        else:
            self.add_output_row("Failed to capture image.")


#############OBB-UPDATES##################
    def process_image(self, image):
        if self.selected_model == "detectron2":
            outputs = self.predictor(image)
            return outputs["instances"].pred_masks.numpy()
        elif self.selected_model == "yolov8-seg":
            return self._process_yolov8(image)
        elif self.selected_model == "yolo-obb":
            return self._process_yolo_obb(image)
        elif self.selected_model == "custom":
            return self._process_yolo_obb(image)
        else:
            self.terminal_frame.insert(tk.END, "No valid model selected\n")
            return np.array([])

    def _process_yolo_obb(self, image):
        if self.obb_model is None:
            self.terminal_frame.insert(tk.END, "OBB model not loaded.\n")
            return []

        self.terminal_frame.insert(tk.END, "Running OBB model on the image...\n")

        results = self.obb_model(image)
        if results:
            self.terminal_frame.insert(tk.END, f"Model produced {len(results)} results.\n")
        else:
            self.terminal_frame.insert(tk.END, "No results produced by the OBB model.\n")

        self.obb_results = results[0] if results else []
        
        annotated_image = self.obb_results.plot() if self.obb_results else image
        self.terminal_frame.insert(tk.END, f"Annotated image processed.\n")

        # Convert annotated image to RGB for display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Display the annotated image
        self.display_captured_image(annotated_image_rgb)

        return annotated_image_rgb

    def _process_yolo_obb_internal_function(self, image):
        if self.obb_model is None:
            self.add_output_row("OBB model not loaded.")
            return []

        results = self.obb_model(image)

        if not results:
            self.add_output_row("No results produced by the OBB model.")
            return []

        image_height, image_width = image.shape[:2]
        conversion_factor_x = self.workspace_width_cm / image_width
        conversion_factor_y = self.workspace_height_cm / image_height

        header = tk.Label(self.output_frame, text="Part No      Xc      Yc      Orientation      Class",
                        bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)

        obb_results_list = []

        try:
            if hasattr(results[0], 'obb') and hasattr(results[0].obb, 'data'):
                obb_data = results[0].obb.data
            elif hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'data'):
                obb_data = results[0].boxes.data
            else:
                raise AttributeError("Unable to find OBB data in results")

            for idx, detection in enumerate(obb_data):
                if len(detection) < 7:
                    continue

                x_center, y_center, width, height, angle, score, class_id = detection[:7].tolist()

                centroid_x, centroid_y = int(x_center), int(y_center)
                cy_transformed = image_height - centroid_y

                real_x = centroid_x * conversion_factor_x
                real_y = cy_transformed * conversion_factor_y

                angle_degrees = angle * 180 / np.pi if angle != 0 else 0

                class_name = results[0].names[int(class_id)] if hasattr(results[0], 'names') else f"Class {int(class_id)}"

                part_no = f"PN{idx+1:02d}"

                row = tk.Label(self.output_frame, text=f"{part_no}      {real_x:.2f}      {real_y:.2f}      {angle_degrees:.0f} degrees      {class_name}",
                            bg="gray", fg="white", font=("Arial", 10))
                row.pack(fill="x", padx=10, pady=2)

                corners = cv2.boxPoints(((x_center, y_center), (width, height), angle_degrees))
                corners = np.int0(corners)
                cv2.drawContours(image, [corners], 0, (0, 255, 0), 2)
                cv2.circle(image, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

                top_right_corner = tuple(corners[1])
                label_x = max(top_right_corner[0] - 10, 0)
                label_y = max(top_right_corner[1] - 10, 0)
                cv2.putText(image, part_no, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                obb_results_list.append((x_center, y_center, width, height, angle, score, class_id))

        except Exception as e:
            self.add_output_row(f"Error processing results: {str(e)}")

        self.display_captured_image(image)

        return obb_results_list
    
    def midpoint(self, p1, p2):
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    def segregation(self):
        if self.captured_image_path is None:
            self.terminal_frame.insert(tk.END, "No image captured for segregation.\n")
            return

        self.captured_image = cv2.imread(self.captured_image_path)
        if self.captured_image is None:
            self.terminal_frame.insert(tk.END, "Failed to load captured image.\n")
            return

        self.clear_output()

        if self.selected_model == "detectron2":
            masks = self.process_image(self.captured_image)
            self.segregation_mask_based(masks)
        elif self.selected_model ==  "yolov8-seg":
            masks = self.process_image(self.captured_image)
            self.segregation_mask_based(masks)
        elif self.selected_model in "yolo-obb":
            obb_results = self.process_image(self.captured_image)
            self.segregation_obb_based(obb_results)
        elif self.selected_model ==  "custom":
            obb_results = self.process_image(self.captured_image)
            self.segregation_obb_based(obb_results)
        else:
            self.terminal_frame.insert(tk.END, "Invalid model selected for segregation.\n")

    def inspect(self):
        if self.captured_image_path is None:
            self.terminal_frame.insert(tk.END, "No image captured for inspection.\n")
            return

        self.captured_image = cv2.imread(self.captured_image_path)
        if self.captured_image is None:
            self.terminal_frame.insert(tk.END, "Failed to load captured image.\n")
            return

        self.clear_output()

        if self.selected_model in ["detectron2", "yolov8-seg"]:
            masks = self.process_image(self.captured_image)
            self.inspect_mask_based(masks)
        elif self.selected_model in ["yolo-obb", "custom"]:
            obb_results = self.process_image(self.captured_image)
            self.inspect_obb_based(obb_results)
        else:
            self.terminal_frame.insert(tk.END, "Invalid model selected for inspection.\n")

    def segregation_mask_based(self, masks):
        if len(masks) == 0:
            self.add_output_row("No object detected for segregation.")
            return

        image_height, image_width = self.captured_image.shape[:2]
        conversion_factor_x = self.workspace_width_cm / image_width
        conversion_factor_y = self.workspace_height_cm / image_height

        header = tk.Label(self.output_frame, text="Part No      Xc      Yc      Orientation      Bin", 
                        bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)

        for idx, mask in enumerate(masks):
            mask_image = (mask * 255).astype('uint8')
            contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = contours[0]
            rect = cv2.minAreaRect(contour)
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cy_transformed = image_height - cy

                real_x = cx * conversion_factor_x
                real_y = cy_transformed * conversion_factor_y

                angle = rect[2]
                if rect[1][0] < rect[1][1]:
                    angle = 90 - angle
                else:
                    angle = 180 - angle

                part_no = f"PN{idx+1:02d}"
                bin_number = self.selected_bin.get()

                row = tk.Label(self.output_frame, text=f"{part_no}      {real_x:.2f}      {real_y:.2f}       {angle:.0f} degrees      {bin_number}",
                            bg="gray", fg="white", font=("Arial", 10))
                row.pack(fill="x", padx=10, pady=2)

                object_data = {"part_no": part_no, "angle": angle, "x": real_x, "y": real_y, "bin": bin_number}
                self.send_to_robot(object_data)

                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(self.captured_image, [box], 0, (0, 255, 0), 2)
                cv2.circle(self.captured_image, (cx, cy), 5, (255, 0, 0), -1)

                top_right_corner = (int(box[1][0]), int(box[1][1]))
                label_x = max(top_right_corner[0] - 10, 0)
                label_y = max(top_right_corner[1] - 10, 0)

                cv2.putText(self.captured_image, part_no, (label_x, label_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                row = tk.Label(self.output_frame, text=f"PN{idx+1:02d}      -      -      -      -",
                            bg="gray", fg="white", font=("Arial", 10))
                row.pack(fill="x", padx=10, pady=2)
                self.add_output_row("No valid object found for segregation.")
        else:
            self.add_output_row(f"No object detected for segregation (Object {idx+1}).")

        self.display_captured_image(self.captured_image)


    def process_image_with_model(self, image):
        
        self.terminal_frame.insert(tk.END, f"Selected model: {self.selected_model}\n")

        if self.selected_model == "obb":
            self.terminal_frame.insert(tk.END, "Processing with OBB model.\n")
            return self._process_yolo_obb(image)
        elif self.selected_model == "custom":
            self.terminal_frame.insert(tk.END, "Processing with Custom model.\n")
            return self._process_custom_model(image)
        elif self.selected_model in ["detectron2", "yolov8-seg"]:
            self.terminal_frame.insert(tk.END, "Processing with segmentation model.\n")
            masks = self.process_image(image)
            if len(masks) == 0:
                self.terminal_frame.insert(tk.END, "No object detected.\n")
                return None
            return masks[0]
        else:
            self.terminal_frame.insert(tk.END, "No valid model selected.\n")
            return None

    def set_model(self, model_name):
       
        valid_models = ["obb", "custom", "detectron2", "yolov8-seg"]
        if model_name in valid_models:
            self.selected_model = model_name
            self.terminal_frame.insert(tk.END, f"Model set to: {model_name}\n")
        else:
            self.terminal_frame.insert(tk.END, "Invalid model selected.\n")

    def initialize_models(self):
        # Initialize OBB model if not already done
        if self.selected_model == "obb" and self.obb_model is None:
            self.terminal_frame.insert(tk.END, "Loading OBB model...\n")
            self.obb_model = self.load_obb_model()  # Example of model loading logic
        elif self.selected_model == "custom" and self.custom_model is None:
            self.terminal_frame.insert(tk.END, "Loading Custom model...\n")
            self.custom_model = self.load_custom_model()  # Example of custom model loading logic
        # Add any other model initialization logic as needed


    def segregation_obb_based(self, obb_results=None):
        # Load the captured image for segregation
        self.captured_image = cv2.imread(self.captured_image_path)
        if self.captured_image is None:
            self.terminal_frame.insert(tk.END, "Failed to load captured image.\n")
            return

        # Clear the output before adding new results
        self.clear_output()

        # Process the image using the selected model if obb_results is not provided
    
        obb_results = self._process_yolo_obb_internal_function(self.captured_image)
     

    
    def inspect_mask_based(self, masks):
        if len(masks) == 0:
            self.add_output_row("No object detected for inspection.")
            return

        limits = self.load_limits_from_settings()
        width_limit = limits.get('width_limit', 10.0)
        height_limit = limits.get('height_limit', 15.0)

        header = tk.Label(self.output_frame, text="Part No      Width      Height      Result",
                        bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)

        for idx, mask in enumerate(masks):
            mask_image = (mask * 255).astype('uint8')
            contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contour = contours[0]
                rect = cv2.minAreaRect(contour)

                object_width_px = rect[1][0]
                object_height_px = rect[1][1]

                conversion_factor_x = self.workspace_width_cm / self.captured_image.shape[1]
                conversion_factor_y = self.workspace_height_cm / self.captured_image.shape[0]

                object_width_cm = object_width_px * conversion_factor_x
                object_height_cm = object_height_px * conversion_factor_y

                part_no = f"PN{idx+1:02d}"

                result = "OKAY" if round(object_width_cm, 2) >= width_limit and round(object_height_cm, 2) >= height_limit else "NOT OKAY"

                row = tk.Label(self.output_frame, text=f"{part_no}      {object_width_cm:.2f}      {object_height_cm:.2f}       {result}",
                            bg="gray", fg="white", font=("Arial", 10))
                row.pack(fill="x", padx=10, pady=2)
                
                data = {
                    "part_no": part_no,
                    "width": object_width_cm,
                    "height": object_height_cm,
                    "result": result
                }
                self.send_inspection_data_to_robot(data)

                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(self.captured_image, [box], 0, (0, 255, 0), 2)

                top_left_corner = tuple(box[1])
                cv2.putText(self.captured_image, part_no, (int(top_left_corner[0]), int(top_left_corner[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.display_captured_image(self.captured_image)

    def inspection_obb_based(self, obb_results):
        # Load the captured image for inspection
        self.captured_image = cv2.imread(self.captured_image_path)
        if self.captured_image is None:
            self.terminal_frame.insert(tk.END, "Failed to load captured image.\n")
            return

        # Clear the output before adding new results
        self.clear_output()

        # Load limits
        limits = self.load_limits_from_settings()
        width_limit = limits.get('width_limit', 10.0)  # Default value
        height_limit = limits.get('height_limit', 15.0)  # Default value

        # Header for the output
        header = tk.Label(self.output_frame, text="Part No      Width      Height      Result",
                        bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)

        for idx, result in enumerate(obb_results):
            x_center, y_center, width, height, angle, score, class_id = result

            # Compute object dimensions
            object_width_px = width
            object_height_px = height

            # Compute conversion factors and object dimensions in cm
            conversion_factor_x = self.workspace_width_cm / self.captured_image.shape[1]
            conversion_factor_y = self.workspace_height_cm / self.captured_image.shape[0]

            object_width_cm = object_width_px * conversion_factor_x
            object_height_cm = object_height_px * conversion_factor_y

            # Generate part number dynamically
            part_no = f"PN{idx + 1:02d}"

            # Set result based on width and height limits
            result_status = "OKAY" if (round(object_width_cm, 2) >= width_limit and round(object_height_cm, 2) >= height_limit) else "NOT OKAY"

            # Display the object's data as a row
            row = tk.Label(self.output_frame, text=f"{part_no}      {object_width_cm:.2f}      {object_height_cm:.2f}       {result_status}",
                        bg="gray", fg="white", font=("Arial", 10))
            row.pack(fill="x", padx=10, pady=2)

            # Send inspection data to the robot
            data = {
                "part_no": part_no,
                "width": object_width_cm,
                "height": object_height_cm,
                "result": result_status
            }
            self.send_inspection_data_to_robot(data)

            # Draw bounding boxes and labels
            box = cv2.boxPoints(((x_center, y_center), (width, height), angle * 180 / np.pi))
            box = np.intp(box)
            cv2.drawContours(self.captured_image, [box], 0, (0, 255, 0), 2)

            # Add part number near the top-left corner of the bounding box
            top_left_corner = tuple(box[1])
            cv2.putText(self.captured_image, part_no, (int(top_left_corner[0]), int(top_left_corner[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the processed image
        self.display_captured_image(self.captured_image)



if __name__ == "__main__":
    root = ctk.CTk()
    app = ObjectDetectionSoftware(root)
    root.mainloop()


                          


