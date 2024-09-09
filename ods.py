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
        
        self.load_settings()
        self.create_menu_bar()
        self.create_main_layout()
        self.create_control_panel()
        self.update_ui_with_settings()

        self.cap = cv2.VideoCapture(self.selected_camera)
        self.update_webcam_feed()
    def setup_models(self):
        self.setup_detectron2()
        self.setup_yolov8()


    def create_menu_bar(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
                                                        
                                                        #file menu
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
        self.model_menu.add_command(label="Detectron2", command=lambda: self.select_model("detectron2"))
        self.model_menu.add_command(label="YOLOv8-Seg", command=lambda: self.select_model("yolov8-seg"))
        self.model_menu.add_command(label="Custom Model", command=self.load_custom_model)
        self.menubar.add_cascade(label="Model", menu=self.model_menu)
                                
        self.limits_menu=tk.Menu(self.menubar,tearoff=0)
        self.limits_menu.add_command(label="Set Limits", command=self.open_limits_dialog)
        self.menubar.add_cascade(label="Limits",menu=self.limits_menu)
          
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
            self.add_output_row("Limits saved successfully.")
        except ValueError:
            self.add_output_row("Invalid input. Please enter numeric values for limits.")
        except Exception as e:
            self.add_output_row(f"Error saving limits: {str(e)}")

# Function to open a directory dialog to choose a model
    def open_model_directory(self):
    # Open file dialog to select either .pt or .h5 files
        model_file = tk.filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", ".pt *.h5"), ("All Files", ".*")]
        )
        if model_file:
            self.add_output_row(f"Custom model selected: {model_file}")

    def select_model(self, model_name):
        self.selected_model = model_name
        self.add_output_row(f"Selected model: {model_name}")

    def load_custom_model(self):
        model_file = tk.filedialog.askopenfilename(
            title="Select Custom Model File",
            filetypes=[("Model Files", "*.pt *.pth"), ("All Files", "*.*")]
        )
        if model_file:
            self.custom_model_path = model_file
            self.selected_model = "custom"
            self.add_output_row(f"Custom model loaded: {model_file}")

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
        self.ods_label = ctk.CTkLabel(self.root, text="Object Detection Software", font=("montserrat", 16,"bold"))
        self.ods_label.place(relx=0.06, rely=0.060)  # Adjust relx and rely to position it directly under AMS-INDIA


        self.webcam_frame = ctk.CTkLabel(self.root, text="", corner_radius=0,bg_color="#EAEDF0")
        self.webcam_frame.place(relx=0.04, rely=0.12, relwidth=0.35, relheight=0.6)

    
        self.saved_image = ctk.CTkLabel(self.root, text="", corner_radius=0,bg_color="#EAEDF0")
        self.saved_image.place(relx=0.405, rely=0.12, relwidth=0.35, relheight=0.6)

        
        self.output_frame = ctk.CTkLabel(self.root, text="OUTPUT",font=("montserrat",14),text_color='black',  corner_radius=0,bg_color="#C0C0C0")
        self.output_frame.place(relx=0.77, rely=0.12, relwidth=0.2, relheight=0.6)

        
        self.bin_selection_label = ctk.CTkLabel(self.root, text="Bin Selection", font=("Arial", 16))
        self.bin_selection_label.place(relx=0.60, rely=0.80, anchor="center")

        
        bin_numbers = [str(i) for i in range(1, 10)]
        self.selected_bin = tk.StringVar(value=bin_numbers[0]) 

        self.bin_option_menu = ctk.CTkOptionMenu(self.root, variable=self.selected_bin, values=bin_numbers)
        self.bin_option_menu.place(relx=0.60, rely=0.82, relwidth=0.08, relheight=0.04)  


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
        self.inputs_frame.place(relx=0.05, rely=0.78, relwidth=0.45, relheight=0.12)

        self.create_input_rows()

        # Add labels to display current limits
    #    self.width_limit_label = ctk.CTkLabel(self.inputs_frame, text=f"Width Limit: {self.width_limit:.2f} cm")
    #    self.width_limit_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

    #    self.height_limit_label = ctk.CTkLabel(self.inputs_frame, text=f"Height Limit: {self.height_limit:.2f} cm")
    #    self.height_limit_label.grid(row=2, column=1, padx=5, pady=5, sticky="w")

    def create_input_rows(self):
        # Workspace Dimension row
  #      self.label3 =ctk.CTkLabel(self.inputs_frame, text="")
  #      self.label3.grid(row=0, column=0, padx=5, pady=5, sticky="w")

  #      self.dummy_entry=ctk.CTkLabel(self.inputs_frame,text="")
  #      self.dummy_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

  #      self.dummy_entry2=ctk.CTkLabel(self.inputs_frame,text="")
  #      self.dummy_entry2.grid(row=0,column=2,padx=5,pady=5 ,sticky="ew")

        self.label1 = ctk.CTkLabel(self.inputs_frame, text="Workspace Dimensions (in CM)")
        self.label1.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.width_entry = ctk.CTkEntry(self.inputs_frame, placeholder_text="Width")
        self.width_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.height_entry = ctk.CTkEntry(self.inputs_frame, placeholder_text="Height")
        self.height_entry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.set_dimension_button = ctk.CTkButton(self.inputs_frame, text="Set Dimension", command=self.set_dimension)
        self.set_dimension_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")


        # Communication row for IP and Port
        self.label2 = ctk.CTkLabel(self.inputs_frame, text="Communication")
        self.label2.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.ip_entry = ctk.CTkEntry(self.inputs_frame, placeholder_text="IP Address")
        self.ip_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.port_entry = ctk.CTkEntry(self.inputs_frame, placeholder_text="Port")
        self.port_entry.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        self.connect_button = ctk.CTkButton(self.inputs_frame, text="Connect", command=self.connect_to_robot)
        self.connect_button.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
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
                self.add_output_row("Invalid input values. Please enter valid numbers.")
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
            self.workspace_width_cm = float(self.width_entry.get())
            self.workspace_height_cm = float(self.height_entry.get())
            self.add_output_row(f"Workspace dimensions set to: {self.workspace_width_cm}cm x {self.workspace_height_cm}cm")
            self.save_settings()
        except ValueError:
            self.add_output_row("Invalid input. Please enter numeric values for width and height.")
    
    def save_settings(self):
        try:
            # Load current settings from file
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    current_settings = json.load(f)
            else:
                current_settings = {}

            # Update workspace dimensions if changed
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

            self.add_output_row("Settings updated and saved successfully.")
        except Exception as e:
            self.add_output_row(f"Error saving settings: {str(e)}")
            print("Exception occurred:", str(e))

        

    def select_camera(self, camera_index):
        self.selected_camera = camera_index
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.selected_camera)
        if not self.cap.isOpened():
            self.add_output_row(f"Error: Could not open camera {self.selected_camera}")
        else:
            self.add_output_row(f"Successfully opened camera {self.selected_camera}")

    def setup_detectron2(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)

    def setup_yolov8(self):
        self.yolo_model = YOLO('yolov8n-seg.pt')

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
                self.add_output_row(f"Workspace dimensions set to: {width}cm x {height}cm")
            except ValueError:
                self.add_output_row("Invalid input. Please enter two numbers separated by a comma.")

    def show_robot_config_dialog(self):
        dialog = ctk.CTkInputDialog(text="Enter robot IP and port (IP,port):", title="Configure Robot")
        config = dialog.get_input()
        if config:
            try:
                ip, port = config.split(',')
                self.robot_ip = ip.strip()
                self.robot_port = int(port.strip())
                self.add_output_row(f"Robot configuration set to: IP={self.robot_ip}, Port={self.robot_port}")
                self.save_settings()  # Save settings after updating
            except ValueError:
                self.add_output_row("Invalid input. Please enter IP and port separated by a comma.")

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
            self.add_output_row(f"Connected to robot at IP={self.robot_ip}, Port={self.robot_port}")

        except ValueError:
            self.add_output_row("Invalid port number. Please enter a valid integer.")
        except socket.error as e:
            self.add_output_row(f"Connection failed: {e}")
        except Exception as e:
            self.add_output_row(f"An unexpected error occurred: {e}")

    def send_to_robot(self, object_data):
        if not self.robot_ip or not self.robot_port:
            # self.output_display.append("Robot IP or port not set. Please configure robot connection.")
            return
        try:
            port = int(self.robot_port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  
                s.connect((self.robot_ip, port))
                message = f"MOVE {object_data['x']:.2f} {object_data['y']:.2f} {object_data['angle']:.2f}"
                s.sendall(message.encode())
                self.output_display.append(f"Sent to robot: {message}")
        except ValueError:
            self.output_display.append("Invalid port number. Enter a valid integer.")
        except Exception as e:
            self.output_display.append(f"Failed to send data to robot: {str(e)}")
        
            
    def send_inspection_data_to_robot(self, inspection_data):
        if not self.robot_ip or not self.robot_port:
            # self.output_display.append("Robot IP or port not set. Please configure robot connection.")
            return
        try:
            port = int(self.robot_port)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  
                s.connect((self.robot_ip, port))
                message = f"Inspection Data: Width: {inspection_data['width']:.2f} cm, Height: {inspection_data['height']:.2f} cm"
                s.sendall(message.encode())
                self.output_display.append(f"Sent to robot: {message}")
        except ValueError:
            self.output_display.append("Invalid port number. Enter a valid integer.")
        except Exception as e:
            self.output_display.append(f"Failed to send data to robot: {str(e)}")


    def capture_and_process_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_image = frame
            masks = self.process_image(self.captured_image)

            if len(masks) == 0:
                self.add_output_row("No object detected.")
                return

            mask = masks[0]
            mask_image = (mask * 255).astype('uint8')
            color_mask = cv2.applyColorMap(mask_image, cv2.COLORMAP_JET)
            segmented_image = cv2.addWeighted(self.captured_image, 0.7, color_mask, 0.3, 0)

            program_directory = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(program_directory, 'output_image.png')
            cv2.imwrite(save_path, segmented_image)
            self.captured_image_path = save_path  # Save path of the processed image

            self.display_captured_image(segmented_image)
        else:
            self.add_output_row("Failed to capture image.")

    def segregation(self):
        if self.captured_image_path is None:
            self.add_output_row("No image captured for segregation.")
            return

        # Load the saved image
        self.captured_image = cv2.imread(self.captured_image_path)

        if self.captured_image is None:
            self.add_output_row("Failed to load captured image.")
            return

        self.clear_output()
        masks = self.process_image(self.captured_image)

        if len(masks) == 0:
            self.add_output_row("No object detected for segregation.")
            return

        # Extract the height and width of the captured image
        image_height, image_width = self.captured_image.shape[:2]

        # Conversion factors from pixels to centimeters based on workspace size
        conversion_factor_x = self.workspace_width_cm / image_width
        conversion_factor_y = self.workspace_height_cm / image_height

        # Add a styled header row (similar to inspect)
        header = tk.Label(self.output_frame, text="Part No      Xc      Yc      Orientation      Bin", 
                        bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)

        for idx, mask in enumerate(masks):
            # Convert the mask into a format suitable for finding contours
            mask_image = (mask * 255).astype('uint8')
            contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour
                contour = contours[0]
                rect = cv2.minAreaRect(contour)
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    # Compute the centroid (position)
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cy_transformed = image_height - cy  # Adjust y-axis direction

                    # Convert centroid to real-world coordinates (in cm)
                    real_x = cx * conversion_factor_x
                    real_y = cy_transformed * conversion_factor_y

                    # Compute the orientation (angle)
                    angle = rect[2]
                    if rect[1][0] < rect[1][1]:
                        angle = 90 - angle
                    else:
                        angle = 180 - angle

                    # Generate part number dynamically for each object
                    part_no = f"PN{idx+1:02d}"

                    # Get the selected bin number
                    bin_number = self.selected_bin.get()

                    # Display the current object's data as a row in the frame, including the bin number
                    row = tk.Label(self.output_frame, text=f"{part_no}      {real_x:.2f} cm      {real_y:.2f} cm      {angle:.0f} degrees      {bin_number}",
                                bg="gray", fg="white", font=("Arial", 10))
                    row.pack(fill="x", padx=10, pady=2)

                    # Store the object data along with bin number for sending to the robot
                    object_data = {"angle": angle, "x": real_x, "y": real_y, "bin": bin_number}

                    # Send the object data to the robot
                    self.send_to_robot(object_data)

                    # Draw the bounding box on the image
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(self.captured_image, [box], 0, (0, 255, 0), 2)

                    # Calculate the top-right corner
                    top_right_corner = (int(box[1][0]), int(box[1][1]))

                    # Adjust label position to be inside the bounding box
                    label_x = top_right_corner[0] - 10  # 10 pixels from the right edge
                    label_y = top_right_corner[1] - 10  # 10 pixels from the top edge

                    # Ensure label position is within image bounds
                    label_x = max(label_x, 0)
                    label_y = max(label_y, 0)

                    # Add the part number label at the top-right corner of the bounding box
                    cv2.putText(self.captured_image, part_no, (label_x, label_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                else:
                    row = tk.Label(self.output_frame, text=f"PN{idx+1:02d}      -      -      -      -",
                                bg="gray", fg="white", font=("Arial", 10))
                    row.pack(fill="x", padx=10, pady=2)
                    self.add_output_row("No valid object found for segregation.")

            else:
                self.add_output_row(f"No object detected for segregation (Object {idx+1}).")

        # Display the image with bounding boxes and labels
        self.display_captured_image(self.captured_image)


    def inspect(self):
        if self.captured_image_path is not None:
            # Load the saved image
            self.captured_image = cv2.imread(self.captured_image_path)

            if self.captured_image is None:
                self.add_output_row("Failed to load captured image.")
                return

            # Clear the output before adding new results
            self.clear_output()

            outputs = self.predictor(self.captured_image)
            masks = outputs["instances"].pred_masks.numpy()

            # Load limits
            limits = self.load_limits_from_settings()
            width_limit = limits.get('width_limit', 10.0)  # Default value
            height_limit = limits.get('height_limit', 15.0)  # Default value

            if len(masks) == 0:
                self.add_output_row("No object detected for inspection.")
                return

            # Add a styled header row (as a Label widget in Tkinter)
            header = tk.Label(self.output_frame, text="Part No      Width      Height      Result",
                            bg="orange", fg="black", font=("Arial", 12, "bold"))
            header.pack(fill="x", padx=10, pady=5)

            # Loop through each detected object (mask)
            for idx, mask in enumerate(masks):
                mask_image = (mask * 255).astype('uint8')
                contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    contour = contours[0]
                    rect = cv2.minAreaRect(contour)

                    object_width_px = rect[1][0]
                    object_height_px = rect[1][1]

                    # Compute conversion factors and object dimensions
                    conversion_factor_x = self.workspace_width_cm / self.captured_image.shape[1]
                    conversion_factor_y = self.workspace_height_cm / self.captured_image.shape[0]

                    object_width_cm = object_width_px * conversion_factor_x
                    object_height_cm = object_height_px * conversion_factor_y

                    # Generate part number dynamically
                    part_no = f"PN{idx+1:02d}"

                    # Set result based on width and height limits
                    if round(object_width_cm, 2) >= width_limit and round(object_height_cm, 2) >= height_limit:
                        result = "OKAY"
                    else:
                        result = "NOT OKAY"

                    # Display the current object's data as a row in the frame
                    row = tk.Label(self.output_frame, text=f"{part_no}      {object_width_cm:.2f} cm      {object_height_cm:.2f} cm      {result}",
                                bg="gray", fg="white", font=("Arial", 10))
                    row.pack(fill="x", padx=10, pady=2)

                    # Draw the bounding box and part number on the image
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(self.captured_image, [box], 0, (0, 255, 0), 2)
                    
                    # Add part number near the top-left corner of the bounding box
                    top_left_corner = tuple(box[1])
                    cv2.putText(self.captured_image, part_no, (int(top_left_corner[1]), int(top_left_corner[2])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the image with bounding boxes and labels
            self.display_captured_image(self.captured_image)

        else:
            self.add_output_row("No image captured for inspection.")

    def add_output_row(self, output_str):
        # Method to add output to the Tkinter output frame
        row = tk.Label(self.output_frame, text=output_str, bg="gray", fg="white", font=("Arial", 10))
        row.pack(fill="x", padx=10, pady=2)
    def process_image(self, image):
        if self.selected_model == "detectron2":
            outputs = self.predictor(image)
            return outputs["instances"].pred_masks.numpy()
        elif self.selected_model == "yolov8":
            results = self.yolo_model(image, stream=True)
            for result in results:
                if result.masks is not None:
                    return result.masks.data.cpu().numpy()
            return np.array([])
        elif self.selected_model == "custom":
            if self.custom_model_path.endswith('.pt'):  # Assuming it's a YOLOv8 model
                custom_model = YOLO(self.custom_model_path)
                results = custom_model(image, stream=True)
                for result in results:
                    if result.masks is not None:
                        return result.masks.data.cpu().numpy()
            else:
                self.add_output_row("Unsupported custom model format")
            return np.array([])
        else:
            self.add_output_row("No valid model selected")
            return np.array([])

    def clear_output(self):
        """Clears all the content from the output frame."""
        for widget in self.output_frame.winfo_children():
               widget.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = ObjectDetectionSoftware(root)
    root.mainloop()

