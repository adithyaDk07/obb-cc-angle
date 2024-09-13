"""HC working on FrontEnd 
- added buttons 
- bins selection toggle 
- annotation buttons 
"""
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
from datetime import datetime
import struct
from pathlib import Path
import threading
import time 
import pandas as pd
from shapely.geometry import Polygon
import math
import random
import subprocess
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
        self.bin_toggle_var = ctk.BooleanVar(value=False)
        self.selected_bin = tk.StringVar(value="1")
        self.setup_models()
        self.workspace_width_cm = 40
        self.workspace_height_cm = 23
        self.robot_socket = None
        self.captured_image = None
        self.captured_image_path = None
        self.robot_ip = ""
        self.robot_port = ""
        
        self.load_settings()
        self.create_menu_bar()
        self.create_main_layout()
        self.create_control_panel()
        self.create_new_control_panel()  
        self.update_ui_with_settings()
        self.processed_results = []
        self.output_display = []
        self.csv_data = None 
        self.save_directory = None
        self.image_counter = 1
        self.json_folder_path = None

        self.object_data = {}
        self.object_dimensions = {}

        self.server_socket = None
        self.receiver_thread = None
        self.settings = self.load_settings_from_file()
        self.setup_image_receiver()
        self.inspection_data_queue = []
        self.inspection_data_lock = threading.Lock()

        self.positive_tolerance = 0.0
        self.negative_tolerance = 0.0

        

    def setup_models(self):
        self.setup_detectron2()
        self.setup_yolov8()


     #########serrver  
 
    def connect_to_robot(self):
        try:
            self.robot_ip = self.ip_entry.get()
            self.robot_port = int(self.port_entry.get())

            if not self.toggle_var.get():
                self.terminal_frame.insert(tk.END, "Please enable the toggle switch to connect.\n")
                return

            # Close existing socket if it exists
            if self.server_socket:
                self.server_socket.close()

            # Create a new socket and start listening
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.robot_ip, self.robot_port))
            self.server_socket.listen(1)

            self.terminal_frame.insert(tk.END, f"Listening for connections at IP={self.robot_ip}, Port={self.robot_port}\n")
            self.terminal_frame.insert(tk.END, "Waiting for incoming connection...\n")

            # Start a new thread to handle the connection
            connection_thread = threading.Thread(target=self.handle_connection, daemon=True)
            connection_thread.start()

            # Save settings if connection setup is successful
            self.save_settings()

        except ValueError:
            self.terminal_frame.insert(tk.END, "Invalid port number. Please enter a valid integer.\n")
        except socket.error as e:
            self.terminal_frame.insert(tk.END, f"Failed to start listening: {e}\n")
        except Exception as e:
            self.terminal_frame.insert(tk.END, f"An unexpected error occurred: {e}\n")


    def handle_connection(self):
        try:
            client_socket, addr = self.server_socket.accept()
            self.terminal_frame.insert(tk.END, f"Connected to {addr}\n")

            # Start receiving messages
            while True:
                message = client_socket.recv(1024).decode('utf-8')
                if not message:
                    break
                self.terminal_frame.insert(tk.END, f"Received: {message}\n")
                self.process_message(message)

        except Exception as e:
            self.terminal_frame.insert(tk.END, f"Connection error: {e}\n")
        finally:
            if 'client_socket' in locals():
                client_socket.close()

    def process_message(self, message):
        # Process the received message here
        # You can add your logic to handle different types of messages
        pass


    def setup_image_receiver(self):
        # This method now just initializes variables, doesn't start the thread
        self.server_socket = None
        self.receiver_thread = None


    def load_settings_from_file(self):
        default_settings = {
            'robot_ip': '192.168.1.100',  # Default values
            'robot_port': 8080  # Default values
        }

        if not os.path.exists(SETTINGS_FILE):
            # Create default settings file if it does not exist
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(default_settings, f, indent=4)

        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                # Ensure that all required keys are present
                for key in default_settings:
                    if key not in settings:
                        settings[key] = default_settings[key]

                return settings
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            # Return default settings in case of an error
            return default_settings
 
 
    def setup_image_receiver(self):
        self.image_receiver_thread = threading.Thread(target=self.receive_images_continuously, daemon=True)
        self.image_receiver_thread.start()


    def receive_images_continuously(self):
        save_dir = Path.home() / "Pictures" / "received-server-img"
        save_dir.mkdir(parents=True, exist_ok=True)

        image_counter = 1

        while True:
            if self.server_socket is None:
                # If the socket is not initialized, wait and continue
                time.sleep(1)
                continue

            try:
                client_socket, addr = self.server_socket.accept()
                self.terminal_frame.insert(tk.END, f"Connected to {addr}\n")
                
                # ... (rest of the image receiving and processing code remains the same)

            except Exception as e:
                self.terminal_frame.insert(tk.END, f"Error receiving image or sending data: {str(e)}\n")
            finally:
                if 'client_socket' in locals():
                    client_socket.close()


    def process_received_image(self, image):
        self.captured_image = image
        self.capture_and_process_image()

    def send_pending_inspection_data(self, client_socket):
        with self.inspection_data_lock:
            while self.inspection_data_queue:
                data = self.inspection_data_queue.pop(0)
                try:
                    json_data = json.dumps(data)
                    client_socket.sendall(json_data.encode())
                    self.terminal_frame.insert(tk.END, f"Sent inspection data: {json_data}\n")
                except Exception as e:
                    self.terminal_frame.insert(tk.END, f"Error sending inspection data: {str(e)}\n")
                    # If sending fails, put the data back in the queue
                    self.inspection_data_queue.insert(0, data)
                    break

    def send_inspection_data_to_robot(self, inspection_data):
        with self.inspection_data_lock:
            self.inspection_data_queue.append(inspection_data)
        self.terminal_frame.insert(tk.END, f"Queued inspection data for sending: {inspection_data}\n")

    """
            UI
    
    """


    def create_new_control_panel(self):
        # Create a new frame for the control panel
        new_control_panel = ctk.CTkFrame(self.root, corner_radius=15)
        new_control_panel.place(relx=0.54, rely=0.74, relwidth=0.2, relheight=0.24)

        # Create the Segregation button on the left side
        segregation_button = ctk.CTkButton(
            new_control_panel, 
            text="Segregation", 
            command=self.segregation, 
            font=("Arial", 16), 
            height=40
        )
        segregation_button.place(relx=0.25, rely=0.2, relwidth=0.4,relheight=0.26, anchor="center")

        # Create the Inspection button on the right side
        inspection_button = ctk.CTkButton(
            new_control_panel, 
            text="Inspection", 
            command=self.inspection, 
            font=("Arial", 16), 
            height=40
        )
        inspection_button.place(relx=0.75, rely=0.2, relwidth=0.4, relheight=0.26 ,anchor="center")

        # Create the Process Image button below the other two buttons
        process_image_button = ctk.CTkButton(
            new_control_panel, 
            text="Process Image", 
            command=self.process_image_button, 
            font=("Arial", 16), 
            height=40
        )
        process_image_button.place(relx=0.5, rely=0.6, relwidth=0.9, relheight=0.24, anchor="center")
 
 
    def create_menu_bar(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
                                                        
                                                        #file menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Upload CSV", command=self.upload_csv)  # Add this line
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
        self.segmentation_menu.add_command(label="Custom SEG", command=self.load_custom_model_seg)

        # Create OBB submenu
        self.obb_menu = tk.Menu(self.model_menu, tearoff=0)
        self.obb_menu.add_command(label="YOLO-OBB", command=lambda: self.select_model("yolo-obb"))
        self.obb_menu.add_command(label="Custom OBB", command= self.load_custom_model_obb)
        

        # Add submenus to the main model menu
        self.model_menu.add_cascade(label="SEGMENTATION", menu=self.segmentation_menu)
        self.model_menu.add_cascade(label="OBB", menu=self.obb_menu)

        # Add the main model menu to the menubar
        self.menubar.add_cascade(label="Model", menu=self.model_menu)


                                
        self.limits_menu=tk.Menu(self.menubar,tearoff=0)
        self.limits_menu.add_command(label="Set Tolerance ", command=self.tolerance_dialog)
        self.menubar.add_cascade(label="Limits",menu=self.limits_menu)
          

    
    def select_model(self, model_name):
        self.selected_model = model_name
        self.terminal_frame.insert(tk.END,f"Selected model: {model_name}")


    
    
    def load_custom_model_seg(self):
        model_file = tk.filedialog.askopenfilename(
            title="Select Custom Model File",
            filetypes=[("Model Files", "*.pt *.pth"), ("All Files", "*.*")]
        )
        if model_file:
            self.custom_model_path = model_file
            self.selected_model = "custom-seg"
            self.terminal_frame.insert(tk.END,f"Custom model loaded: {model_file}")

   
    def load_custom_model_obb(self):
        model_file = tk.filedialog.askopenfilename(
            title="Select Custom Model File",
            filetypes=[("Model Files", "*.pt *.pth"), ("All Files", "*.*")]
        )
        if model_file:
            self.custom_model_path = model_file
            try:
                self.custom_model = YOLO(self.custom_model_path)
                self.selected_model = "custom-obb"
                self.terminal_frame.insert(tk.END, f"Custom model loaded: {model_file}\n")
            except Exception as e:
                self.terminal_frame.insert(tk.END, f"Error loading custom model: {str(e)}\n")


    

    def create_main_layout(self):
        # Logo and title
        logo_image = Image.open(r"/home/adithyadk/Desktop/checkerBoard/primary.jpg")
        logo_image = logo_image.resize((80, 80), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(logo_image)

        self.logo_label = ctk.CTkLabel(self.root, image=self.logo_photo, text="")
        self.logo_label.place(relx=0.01, rely=0.02)

        self.ams_label = ctk.CTkLabel(self.root, text="AMS-INDIA", font=("Montserrat", 24, "bold"))
        self.ams_label.place(relx=0.06, rely=0.028)

        self.ods_label = ctk.CTkLabel(self.root, text="Object Detection Software", font=("Montserrat", 16, "bold"))
        self.ods_label.place(relx=0.06, rely=0.060)

        # Main image display
        self.saved_image = ctk.CTkLabel(self.root, text="", corner_radius=15, bg_color="#EAEDF0")
        self.saved_image.place(relx=0.02, rely=0.12, relwidth=0.5, relheight=0.6)

        self.output_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.output_frame.place(relx=0.54, rely=0.12, relwidth=0.2, relheight=0.6)
        
        output_title = ctk.CTkLabel(self.output_frame, text="OUTPUT", font=("Montserrat", 14, "bold"), text_color='white')
        output_title.pack(pady=10)

        # Create a frame to hold the text widget and scrollbars
        text_frame = ctk.CTkFrame(self.output_frame)
        text_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Create vertical scrollbar
        v_scrollbar = ctk.CTkScrollbar(text_frame, orientation="vertical")
        v_scrollbar.pack(side="right", fill="y")

        # Create horizontal scrollbar
        h_scrollbar = ctk.CTkScrollbar(text_frame, orientation="horizontal")
        h_scrollbar.pack(side="bottom", fill="x")

        # Create the text widget with both scrollbars
        self.output_text = ctk.CTkTextbox(text_frame, font=("Arial", 12), text_color='white', corner_radius=10,
                                          yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set, wrap="none")
        self.output_text.pack(expand=True, fill="both")

        # Configure the scrollbars
        v_scrollbar.configure(command=self.output_text.yview)
        h_scrollbar.configure(command=self.output_text.xview)

      
        self.terminal_frame = ctk.CTkTextbox(self.root, font=("Arial", 14), text_color='white', corner_radius=15)
        self.terminal_frame.place(relx=0.02, rely=0.74, relwidth=0.5, relheight=0.24)



    def create_control_panel(self):
        # Control panel frame
        control_panel = ctk.CTkFrame(self.root, corner_radius=15)
        control_panel.place(relx=0.77, rely=0.12, relwidth=0.2, relheight=0.84)

        # Annotation frame
        self.annotation_frame = ctk.CTkFrame(control_panel, corner_radius=10)
        self.annotation_frame.pack(padx=10, pady=10, fill="x")

        self.annotation_label = ctk.CTkLabel(self.annotation_frame, text="Annotation", font=("Arial", 14, "bold"))
        self.annotation_label.pack(pady=5)

        self.seg_annotation_button = ctk.CTkButton(self.annotation_frame, text="Segmentation Annotation",
                                                   command=self.segmentation_annotate)
        self.seg_annotation_button.pack(padx=10, pady=5, fill="x")

        self.obb_annotation_button = ctk.CTkButton(self.annotation_frame, text="I OBB Annotation",
                                                   command=self.obb_annotate)
        self.obb_annotation_button.pack(padx=10, pady=5, fill="x")

        self.obb_convert_button = ctk.CTkButton(self.annotation_frame, text="II OBB Converter",
                                                command=self.load_json_folder)
        self.obb_convert_button.pack(padx=10, pady=5, fill="x")
                                                

        # Train mode switch
        self.togg2_var = ctk.BooleanVar(value=False)
        self.togg2_switch = ctk.CTkSwitch(control_panel, text="TRAIN MODE", variable=self.togg2_var, command=self.toggle_capture)
        self.togg2_switch.pack(padx=10, pady=10, fill="x")

        # Capture button
        self.capture_button = ctk.CTkButton(control_panel, text="Capture", command=self.capture_image)
        self.capture_button.pack(padx=10, pady=5, fill="x")
        self.capture_button.configure(state="disabled")  # Initially disabled

        # Load image button
        self.load_image_button = ctk.CTkButton(control_panel, text="Load Image", command=self.load_image)
        self.load_image_button.pack(padx=10, pady=5, fill="x")

        

        # Robot Configuration
        robot_config_frame = ctk.CTkFrame(control_panel, corner_radius=10)
        robot_config_frame.pack(padx=10, pady=10, fill="x")

        self.robot_config_label = ctk.CTkLabel(robot_config_frame, text="Robot Configuration", font=("Arial", 14, "bold"))
        self.robot_config_label.pack(pady=5)

        self.toggle_var = ctk.BooleanVar(value=False)
        self.toggle_switch = ctk.CTkSwitch(robot_config_frame, text="Enable", variable=self.toggle_var, command=self.toggle_inputs)
        self.toggle_switch.pack(padx=10, pady=5, fill="x")

        self.inputs_frame = ctk.CTkFrame(robot_config_frame, corner_radius=10)
        self.inputs_frame.pack(padx=10, pady=5, fill="x")

        self.bin_selection_frame = ctk.CTkFrame(control_panel, corner_radius=10)
        self.bin_selection_frame.pack(padx=10, pady=10, fill="x")

        self.bin_selection_label = ctk.CTkLabel(self.bin_selection_frame, text="Bin Selection", font=("Arial", 14, "bold"))
        self.bin_selection_label.pack(pady=(5, 0))

        self.bin_toggle_switch = ctk.CTkSwitch(self.bin_selection_frame, text="Enable Bin Selection", 
                                               variable=self.bin_toggle_var, command=self.toggle_bin_selection)
        self.bin_toggle_switch.pack(padx=10, pady=5, fill="x")

        bin_numbers = [str(i) for i in range(1, 10)]
        self.bin_option_menu = ctk.CTkOptionMenu(self.bin_selection_frame, variable=self.selected_bin, values=bin_numbers)
        self.bin_option_menu.pack(padx=10, pady=5, fill="x")
        self.bin_option_menu.configure(state="disabled")  # Initially disabled
        
        

        self.create_input_rows()

    def toggle_bin_selection(self):
        if self.bin_toggle_var.get():
            self.bin_option_menu.configure(state="normal")
            self.terminal_frame.insert(tk.END, "Bin selection enabled.\n")
        else:
            self.bin_option_menu.configure(state="disabled")
            self.terminal_frame.insert(tk.END, "Bin selection disabled.\n")

#################3333
    def segmentation_annotate(self):
        try:
            labelme_path = r"/home/adithyadk/Desktop/checkerBoard/labelme/labelme/__main__.py"
            subprocess.Popen(["python", labelme_path])
            self.terminal_frame.insert(tk.END, "LabelMe annotation tool launched successfully.\n")
        except Exception as e:
            self.terminal_frame.insert(tk.END, f"Error launching LabelMe: {str(e)}\n")

    def obb_annotate(self):
        try:
            labelme_path = r"/home/adithyadk/Desktop/checkerBoard/labelme/labelme/__main__.py"
            subprocess.Popen(["python", labelme_path])
            self.terminal_frame.insert(tk.END, "LabelMe annotation tool launched successfully.\n")
        except Exception as e:
            self.terminal_frame.insert(tk.END, f"Error launching LabelMe: {str(e)}\n")

   

    def toggle_capture(self):
        if self.togg2_var.get():
            self.capture_button.configure(state="normal")
        else:
            self.capture_button.configure(state="disabled")

#################3333

    
    def capture_image(self):
        if self.togg2_var.get():
            ret, frame = self.cap.read()
            if ret:
                
                    # If it's the first capture, ask for save location
                    if self.save_directory is None:
                        self.save_directory = tk.filedialog.askdirectory(title="Select Directory to Save Images")
                        if not self.save_directory:  # If user cancels directory selection
                            self.terminal_frame.insert(tk.END, "Image capture cancelled. No directory selected.\n")
                            return

                    # Generate filename
                    filename = f"image_{self.image_counter}.png"
                    filepath = os.path.join(self.save_directory, filename)
                    
                    # Save the image
                    cv2.imwrite(filepath, frame)
                    self.terminal_frame.insert(tk.END, f"Image saved as {filepath}\n")
                    
                    # Increment the counter for next image
                    self.image_counter += 1
                    
                    # Display the captured image
                    self.display_captured_image(frame)
            else:
                self.terminal_frame.insert(tk.END, "Failed to capture image.\n")


    
    def train(self):
        # Placeholder for train functionality
        self.terminal_frame.insert(tk.END, "Training functionality not implemented yet.\n")

    def annotate(self):
        # Placeholder for annotate functionality
        self.terminal_frame.insert(tk.END, "Annotation functionality not implemented yet.\n")

    # ... (rest of the class methods remain the same)

    def update_terminal(self, text):
        self.terminal_frame.insert(tk.END, f"{text}\n")
        self.terminal_frame.see(tk.END)  # Scroll to the end for the latest log

    def create_input_rows(self):
        # Workspace Dimensions
        ctk.CTkLabel(self.inputs_frame, text="Workspace Dimensions (cm)", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        ctk.CTkLabel(self.inputs_frame, text="Width:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.width_entry = ctk.CTkEntry(self.inputs_frame)
        self.width_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        ctk.CTkLabel(self.inputs_frame, text="Height:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.height_entry = ctk.CTkEntry(self.inputs_frame)
        self.height_entry.grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        self.set_dimension_button = ctk.CTkButton(self.inputs_frame, text="Set Dimension", command=self.set_dimension)
        self.set_dimension_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Communication
        ctk.CTkLabel(self.inputs_frame, text="Communication", font=("Arial", 12, "bold")).grid(row=4, column=0, columnspan=2, padx=5, pady=(10,5), sticky="w")

        ctk.CTkLabel(self.inputs_frame, text="IP Address:").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.ip_entry = ctk.CTkEntry(self.inputs_frame)
        self.ip_entry.grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        ctk.CTkLabel(self.inputs_frame, text="Port:").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.port_entry = ctk.CTkEntry(self.inputs_frame)
        self.port_entry.grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        self.connect_button = ctk.CTkButton(self.inputs_frame, text="Connect", command=self.connect_to_robot)
        self.connect_button.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        for i in range(2):
            self.inputs_frame.grid_columnconfigure(i, weight=1)

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

    def select_camera(self, camera_index):
        self.selected_camera = camera_index
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.selected_camera)
        if not self.cap.isOpened():
            self.terminal_frame.insert(tk.END,f"Error: Could not open camera {self.selected_camera}\n")
        else:
            self.terminal_frame.insert(tk.END,f"Successfully opened camera {self.selected_camera}\n")


    def load_json_folder(self):
        folder_path = tk.filedialog.askdirectory(title="Select Folder with JSON Files")

        if not folder_path:
            self.terminal_frame.insert(tk.END,"No folder selected.\n")
            return

        self.json_folder_path = folder_path
        self.terminal_frame.insert(tk.END,f"Folder selected: {self.json_folder_path}\n")

        json_files = [f for f in os.listdir(self.json_folder_path) if f.endswith('.json')]

        if not json_files:
            self.terminal_frame.insert(tk.END,"No JSON files found in the selected folder.\n")
            return

        self.terminal_frame.insert(tk.END,f"Found {len(json_files)} JSON files.\n")
        self.process_json_files()

    def process_json_files(self):
        if not self.json_folder_path:
            self.terminal_frame.insert(tk.END,"JSON folder path not set. Please load the folder first.\n")
            return

        self.terminal_frame.insert(tk.END,f"Processing JSON files from folder: {self.json_folder_path}\n")

        json_files = [f for f in os.listdir(self.json_folder_path) if f.endswith('.json')]

        if not json_files:
            self.terminal_frame.insert(tk.END,"No JSON files found in the folder.\n")
            return

        output_dir = os.path.join(self.json_folder_path, "LABEL_TXT")
        os.makedirs(output_dir, exist_ok=True)

        all_labels = set()

        for json_file in json_files:
            file_path = os.path.join(self.json_folder_path, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                yolo_annotations, labels = self.convert_to_yolo_obb(data)
                all_labels.update(labels)

                output_filename = os.path.splitext(json_file)[0] + '.txt'
                output_path = os.path.join(output_dir, output_filename)

                with open(output_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))

                self.terminal_frame.insert(tk.END,f"Processed {json_file} -> {output_filename}\n")
            except Exception as e:
                self.terminal_frame.insert(tk.END,f"Error processing {json_file}: {str(e)}\n")

        # Write class names to a file
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            f.write('\n'.join(sorted(all_labels)))
        self.terminal_frame.insert(tk.END,f"CONVERSION PROCESS BEEN COMPLETED. PROCEED TO MODEL TRAINING")
        self.update_terminal(f"Class names written to {os.path.join(output_dir, 'classes.txt')}")

    @staticmethod
    def calculate_obb(points):
        polygon = Polygon(points)
        min_rect = polygon.minimum_rotated_rectangle
        coords = list(min_rect.exterior.coords)[:-1]

        center_x = sum(x for x, y in coords) / 4
        center_y = sum(y for x, y in coords) / 4

        width = math.sqrt((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)
        height = math.sqrt((coords[1][0] - coords[2][0])**2 + (coords[1][1] - coords[2][1])**2)

        if height > width:
            width, height = height, width

        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        angle = math.atan2(dy, dx)

        return center_x, center_y, width, height, angle

    def convert_to_yolo_obb(self, data):
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        yolo_annotations = []
        labels = []

        for shape in data['shapes']:
            label = shape['label']
            if label not in labels:
                labels.append(label)
            class_id = labels.index(label)

            points = shape['points']
            center_x, center_y, width, height, angle = self.calculate_obb(points)

            # Normalize coordinates
            center_x /= image_width
            center_y /= image_height
            width /= image_width
            height /= image_height

            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {angle:.6f}")

        return yolo_annotations, labels

    def update_terminal(self, text):
        # Assuming this method exists in your class to update the UI
        print(text)  # For debugging purposes


  


    """INITIALIZATION OF MODELS"""
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
    """END INITIALIZATION OF MODELS"""

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
         


    def add_output_row(self, output_str):
        # Method to add output to the Tkinter output frame
        row = tk.Label(self.output_frame, text=output_str, bg="gray", fg="white", font=("Arial", 10))
        row.pack(fill="x", padx=10, pady=2)
  
  
    def add_terminal_row(self, output_str):
        self.terminal_frame.configure(text=self.terminal_frame.cget("text") + output_str + "\n")
   
   
   
   
    """ANNOTATE CONNVERTER"""
   
    def json_to_yolo_obb(self):
        # Get the user's home directory
        home_dir = os.path.expanduser("~")
        
        # Define the input and output directories
        input_dir = os.path.join(home_dir, "Documents", "AMS", "FILES", "IMAGES", "CAPTURED IMAGE", "JSON")
        output_dir = os.path.join(home_dir, "Documents", "AMS", "FILES", "IMAGES", "CAPTURED IMAGE", "TXT")
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if the input directory exists
        if not os.path.exists(input_dir):
            self.terminal_frame.insert(tk.END,f"Error: Input directory does not exist: {input_dir}\n")
            return
        
        # Loop through all JSON files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                json_path = os.path.join(input_dir, filename)
                output_txt_path = os.path.join(output_dir, filename.replace(".json", ".txt"))
                
                self.terminal_frame.insert(tk.END,f"Processing {json_path}...\n")
                try:
                    # Load the LabelMe JSON file
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                except Exception as e:
                    print(f"Error loading JSON file: {e}")
                    continue
                
                # Extract image width and height from the JSON
                image_width = data.get('imageWidth')
                image_height = data.get('imageHeight')
                
                if image_width is None or image_height is None:
                    self.terminal_frame.insert(tk.END,f"Error: Image dimensions are missing from {json_path}. Skipping...\n")
                    continue
                
                annotations = data.get('shapes', [])
            
                
                # Optional: Create a label map for numeric class IDs
                label_map = {label: idx for idx, label in enumerate(set(shape['label'] for shape in annotations))}
                
                # Write the output file
                with open(output_txt_path, 'w') as output_file:
                    for shape in annotations:
                        points = shape.get('points')
                        if points is None:
                            self.terminal_frame.insert(tk.END,"Skipping shape with missing points.\n")
                            continue
                        
                        label = shape.get('label')
                        if label not in label_map:
                            self.terminal_frame.insert(tk.END,f"Skipping shape with unknown label: {label}\n")
                            continue
                        
                        # Create a polygon from the points
                        polygon = Polygon(points)
                        
                        # Calculate the minimum oriented bounding box (OBB)
                        obb = cv2.minAreaRect(np.array(points).astype(np.float32))
                        
                        # Get the center (x, y), (width, height), and angle of rotation of the OBB
                        (center_x, center_y), (width, height), angle = obb
                        # Normalize the coordinates (divide by image size)
                        center_x /= image_width
                        center_y /= image_height
                        width /= image_width
                        height /= image_height
                        # Convert angle from degrees to radians
                        angle_radians = math.radians(angle)
                        
                        # Write to YOLO-OBB format: label (numeric ID), center_x, center_y, width, height, angle (in radians)
                        output_file.write(f"{label_map[label]} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {angle_radians:.6f}\n")
                
                self.terminal_frame.insert(tk.END,f"Finished processing {json_path}. Output saved to {output_txt_path}\n")

        self.terminal_frame.insert(tk.END,"Conversion completed.\n")
   
   
   
   
    """LOADINS AND SAVING THE SETTINGS"""
    def load_settings(self):
        default_settings = {
            'workspace_width_cm': 40.0,
            'workspace_height_cm': 23.0,
            'robot_ip': '',
            'robot_port': 0,
            'width_limit': 10.0,
            'height_limit': 15.0,
            'custom_model_seg_path': '',
            'custom_model_obb_path': '',
            'csv_file_path': ''
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
            
            # Load custom model paths and CSV file path
            self.custom_model_seg_path = settings.get('custom_model_seg_path', '')
            self.custom_model_obb_path = settings.get('custom_model_obb_path', '')
            self.csv_file_path = settings.get('csv_file_path', '')
            
            # Check if files exist and print message if missing
            self._check_file_exists(self.custom_model_seg_path, "Custom segmentation model")
            self._check_file_exists(self.custom_model_obb_path, "Custom OBB model")
            self._check_file_exists(self.csv_file_path, "CSV file")
            
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            # Use default settings if there's an error
            self.__dict__.update(default_settings)

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
            
            # Save custom model paths and CSV file path
            current_settings['custom_model_seg_path'] = self.custom_model_seg_path
            current_settings['custom_model_obb_path'] = self.custom_model_obb_path
            current_settings['csv_file_path'] = self.csv_file_path
            
            # Save updated settings back to file
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(current_settings, f, indent=4)
            
            self.terminal_frame.insert(tk.END, "Settings updated and saved successfully.\n")
        except Exception as e:
            self.terminal_frame.insert(tk.END, f"Error saving settings: {str(e)}\n")
            print("Exception occurred:", str(e))



    def _check_file_exists(self, file_path, file_description):
        if file_path and not os.path.exists(file_path):
            self.terminal_frame.insert(tk.END, f"{file_description} is missing: {file_path}\n")


    """IMAGE CAPTURING PROCESS"""
    def load_image(self):
        file_path = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            self.captured_image = cv2.imread(file_path)
            self.captured_image_path = file_path
            self.display_captured_image(self.captured_image)
            self.terminal_frame.insert(tk.END, f"Image loaded: {file_path}\n")



    def display_captured_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        self.saved_image.configure(image=photo)
        self.saved_image.image = photo

    """CHECKING TOLERANCE VALUE"""

    def tolerance_dialog(self):
        # Open a dialog to set tolerance values
        tolerance_window = tk.Toplevel(self.root)
        tolerance_window.title("Set Tolerance")
        tolerance_window.geometry("300x150")

        # Create label and entry for positive tolerance
        tk.Label(tolerance_window, text="Positive Tolerance (+):").grid(row=0, column=0, padx=10, pady=10)
        positive_entry = tk.Entry(tolerance_window)
        positive_entry.grid(row=0, column=1, padx=10)

        # Create label and entry for negative tolerance
        tk.Label(tolerance_window, text="Negative Tolerance (-):").grid(row=1, column=0, padx=10, pady=10)
        negative_entry = tk.Entry(tolerance_window)
        negative_entry.grid(row=1, column=1, padx=10)

        def save_tolerance():
            try:
                # Save the tolerance values
                self.positive_tolerance = float(positive_entry.get())
                self.negative_tolerance = float(negative_entry.get())
                tolerance_window.destroy()  # Close the window after saving
            except ValueError:
             self.terminal_frame.insert(tk.END,"Invalid Input", "Please enter valid numeric values.\n")

        # Submit button to save tolerance values
        submit_button = tk.Button(tolerance_window, text="Save", command=save_tolerance)
        submit_button.grid(row=2, column=0, columnspan=2, pady=20)

    def check_tolerance(self, measured_value, csv_value):
        # Check if the measured value is within tolerance range
        tolerance_high = csv_value + self.positive_tolerance
        tolerance_low = csv_value - self.negative_tolerance

        return "OK" if tolerance_low <= measured_value <= tolerance_high else "NOT OK"
  
  
  
    """CSV IMPORT"""
    def upload_csv(self):
        file_path = tk.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.csv_data = pd.read_csv(file_path)
                self.terminal_frame.insert(tk.END, f"CSV file uploaded and processed: {file_path}\n")
            except Exception as e:
                self.terminal_frame.insert(tk.END, f"Error processing CSV file: {str(e)}\n")    


    """MODEL SELECTION FUNCTIONS"""
    """                         SEGREGATE FUNCTIONS"""
    """                                             INSPECTION FUNCTION"""

    def process_image_button(self):
        if self.captured_image is not None:
            self.process_image(self.captured_image)  # Process the captured image
            self.terminal_frame.insert(tk.END,"IMAGE PROCESSING HAS BEEN DONE PROCEED TO FURTHER SEGREGATION / INSPECTION \n")
        else:
            self.terminal_frame.insert(tk.END, "No image available to process\n")
    def process_image(self, image):
        if self.selected_model == "detectron2":
            return self._process_detectron2(image)
        elif self.selected_model == "yolov8-seg":
            return self._process_yolov8(image)
        elif self.selected_model == "custom-seg":
            return self._process_yolov8(image)
        elif self.selected_model == "yolo-obb":
            self.terminal_frame.insert(tk.END,"IMAGE PROCESSING HAS BEEN DONE PROCEED TO FURTHER SEGREGATION/INSPECTION")
            return self.process_obb(image)
        elif self.selected_model == "custom-obb":
            return self.process_obb(image)
        else:
            self.terminal_frame.insert(tk.END, "No valid model selected\n")
            return np.array([])
 

    """
        detectron2 process done here
    """
    def _process_detectron2(self, image):
        outputs = self.predictor(image)
        masks = outputs["instances"].pred_masks.cpu().numpy()
        masks = self.process_image(self.captured_image)

        if masks.size == 0:
            self.terminal_frame.insert(tk.END, "No objects detected.\n")
            return

        # Clear previous output and results
        self.clear_output()
        self.processed_results = []

        # Process each detected object
        for i, mask in enumerate(masks):
            # Calculate object properties
            orientation, centroid, height, width = self.calculate_object_properties(mask)

            # Convert pixel dimensions to real-world dimensions
            real_height = height * self.workspace_height_cm / self.captured_image.shape[0]
            real_width = width * self.workspace_width_cm / self.captured_image.shape[1]
            real_x = centroid[0] * self.workspace_width_cm / self.captured_image.shape[1]
            real_y = centroid[1] * self.workspace_height_cm / self.captured_image.shape[0]

            # Store results
            self.processed_results.append({
                'id': i + 1,
                'orientation': orientation,
                'centroid': (real_x, real_y),
                'height': real_height,
                'width': real_width
            })

            # Display results
            result = f"Object {i+1}:\n"
            result += f"  Orientation: {orientation:.2f} degrees\n"
            result += f"  Centroid: ({real_x:.2f} cm, {real_y:.2f} cm)\n"
            result += f"  Height: {real_height:.2f} cm\n"
            result += f"  Width: {real_width:.2f} cm\n"

            self.terminal_frame.insert(tk.END, result + "\n")
    """
        data retrieve for mask-SEGMENT
    """

    def calculate_object_properties(self, mask):
     
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, (0, 0), 0, 0  

        contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(contour)
        orientation = rect[2]  

     
        if rect[1][0] < rect[1][1]:
            orientation = 90 - orientation
        else:
            orientation = 180 - orientation

       
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            centroid = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
        else:
            centroid = (0, 0) 

   
        x, y, width, height = cv2.boundingRect(contour)

        return orientation, centroid, height, width

    """
        yolo-seg-masking operation done here
    """
    def _process_yolov8(self, image):
        if not hasattr(self, 'yolo_model'):
            self.terminal_frame.insert(tk.END, "YOLOv8 model not initialized\n")
            return

        # Process the image with the YOLOv8 model
        results = self.yolo_model(image, stream=True)
        masks = None

        # Extract masks from the results
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                break

        if masks is None or masks.size == 0:
            self.terminal_frame.insert(tk.END, "No objects detected.\n")
            return

        # Clear previous output and results
        self.clear_output()
        self.processed_results = []

        # Process each detected object
        for i, mask in enumerate(masks):
            # Calculate object properties
            orientation, centroid, height, width = self.calculate_object_properties(mask)

            # Convert pixel dimensions to real-world dimensions
            real_height = height * self.workspace_height_cm / image.shape[0]
            real_width = width * self.workspace_width_cm / image.shape[1]
            real_x = centroid[0] * self.workspace_width_cm / image.shape[1]
            real_y = centroid[1] * self.workspace_height_cm / image.shape[0]

            # Store results
            self.processed_results.append({
                'id': i + 1,
                'orientation': orientation,
                'centroid': (real_x, real_y),
                'height': real_height,
                'width': real_width
            })

            # Display results
            result = f"Object {i+1}:\n"
            result += f"  Orientation: {orientation:.2f} degrees\n"
            result += f"  Centroid: ({real_x:.2f} cm, {real_y:.2f} cm)\n"
            result += f"  Height: {real_height:.2f} cm\n"
            result += f"  Width: {real_width:.2f} cm\n"

            self.terminal_frame.insert(tk.END, result + "\n")

        
    """
        yolo-obb-BOX operation done here also  for custom-obb 
    """
    
    def process_obb(self, image):
        if self.custom_model is None:
            return "OBB model not loaded."

        results = self.custom_model(image)

        if not results:
            return "No results produced by the OBB model."

        image_height, image_width = image.shape[:2]
        conversion_factor_x = self.workspace_width_cm / image_width
        conversion_factor_y = self.workspace_height_cm / image_height

        try:
            if hasattr(results[0], 'obb') and hasattr(results[0].obb, 'data'):
                obb_data = results[0].obb.data
            elif hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'data'):
                obb_data = results[0].boxes.data
            else:
                raise AttributeError("Unable to find OBB data in results")

            # Filter and sort detections
            filtered_detections = []
            for detection in obb_data:
                if len(detection) < 7:
                    continue
                x_center, y_center, width, height, angle, score, class_id = detection[:7].tolist()

                if score < 0.7:
                    continue

                if width < 0.05 * image_width or height < 0.05 * image_height:
                    continue

                filtered_detections.append((score, detection))

            filtered_detections.sort(reverse=True, key=lambda x: x[0])

            self.object_dimensions.clear()
            self.object_data.clear()

            if filtered_detections:
                # Set the color to green for all elements (BGR format)
                color = (0, 255, 0)  # Green in BGR

                for idx, (_, top_detection) in enumerate(filtered_detections):
                    x_center, y_center, width, height, angle, score, class_id = top_detection[:7].tolist()

                    centroid_x, centroid_y = int(x_center), int(y_center)
                    cy_transformed = image_height - centroid_y

                    real_x = centroid_x * conversion_factor_x
                    real_y = cy_transformed * conversion_factor_y

                    angle_degrees = angle * 180 / np.pi if angle != 0 else 0

                    class_name = results[0].names[int(class_id)] if hasattr(results[0], 'names') else f"Class {int(class_id)}"

                    part_no = f"PN{idx+1:02d}"  # Generate part number

                    # Save data in dictionaries
                    self.object_dimensions[part_no] = {
                        "width": width * conversion_factor_x,
                        "height": height * conversion_factor_y,
                        "x_center": real_x,
                        "y_center": real_y,
                        "angle": angle_degrees,
                        "class_id": int(class_id),
                        "class_name": class_name
                    }

                    self.object_data[part_no] = {
                        "x_center": real_x,
                        "y_center": real_y,
                        "angle": angle_degrees,
                        "class_id": int(class_id),
                        "class_name": class_name
                    }

                    # Draw rotated bounding box
                    rect = ((x_center, y_center), (width, height), angle * 180 / np.pi)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # Draw the bounding box with green color for each edge
                    cv2.line(image, tuple(box[0]), tuple(box[1]), color, 2)
                    cv2.line(image, tuple(box[1]), tuple(box[2]), color, 2)
                    cv2.line(image, tuple(box[2]), tuple(box[3]), color, 2)
                    cv2.line(image, tuple(box[3]), tuple(box[0]), color, 2)

                    # Draw centroid
                    cv2.circle(image, (centroid_x, centroid_y), 3, color, -1)

                    # Add labels with part number and class name
                    label = f"{part_no} ({class_name}): {score:.2f} {angle_degrees:.2f}°"
                    self.draw_text_with_background(image, label, (centroid_x, centroid_y - 5), color, (255, 255, 255))

                    # Add centroid coordinates
                    centroid_label = f"({centroid_x}, {centroid_y})"
                    cv2.putText(image, centroid_label, (centroid_x - 40, centroid_y + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            self.display_results(image)
            return f"Total objects detected: {len(self.object_data)}"

        except Exception as e:
            return f"Error processing results: {str(e)}"


    def midpoint(self, p1, p2):
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

  
    def draw_text_with_background(self, image, text, position, bg_color, text_color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Draw background rectangle
        cv2.rectangle(image, 
                    (position[0], position[1] - text_size[1] - 5), 
                    (position[0] + text_size[0], position[1] + 5), 
                    bg_color, 
                    -1)

        # Draw text
        cv2.putText(image, text, position, font, font_scale, text_color, thickness)
    

    """
        SEGREGATION-MASK-BUTTON
    """
    def segregation_mask(self):
        if not self.processed_results:
            self.terminal_frame.insert(tk.END, "No processed data available. Please capture and process an image first.\n")
            return

        header = tk.Label(self.output_frame, text="Part No      Xc      Yc      Orientation     ", 
                        bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)
        for result in self.processed_results:

            row = tk.Label(self.output_frame, text=f"{result['id']}      {result['centroid'][0]:.2f} cm      {result['centroid'][1]:.2f}      {result['orientation']:.2f} degrees     ",
                                bg="gray", fg="white", font=("Arial", 10))
            row.pack(fill="x", padx=10, pady=2)

        self.terminal_frame.insert(tk.END, "Segregation data displayed in the output panel.\n")


    """
        SEGREGATION-MASK-BUTTON
    """
    def segregation(self):
        if self.selected_model == "detectron2":
            self.segregation_mask()
        elif self.selected_model=="custom-seg":
            self.segregation_mask()
        elif self.selected_model=="yolov8-seg":
            self.segregation_mask()
        elif self.selected_model=="yolo-obb":
            self.get_object_data()
        elif self.selected_model=="custom-obb":
            self.get_object_data()
        else:
            self.terminal_frame.insert(tk.END, "No valid model selected. Hence no Segregation data\n")

    """
        SEGREGATION-OBB-BUTTON
    """
    def get_object_data(self):
        # Clear previous content in the output frame
        self.clear_output()

        # Add a styled header row
        header = tk.Label(self.output_frame, text="Part No      Xc      Yc      Orientation      Bin", 
                          bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)

        # Get the selected bin number
        bin_number = self.selected_bin.get()

        # Create a copy of the captured image for drawing
        display_image = self.captured_image.copy()

        for part_no, data in self.object_data.items():
            # Display the current object's data as a row in the frame
            row = tk.Label(self.output_frame, 
                           text=f"{part_no}      {data['x_center']:.2f}      {data['y_center']:.2f}       {data['angle']:.0f} degrees      {bin_number}",
                           bg="gray", fg="white", font=("Arial", 10))
            row.pack(fill="x", padx=10, pady=2)

            # Prepare object data for sending to the robot
            object_data = {
                "part_no": part_no,
                "angle": data['angle'],
                "x": data['x_center'],
                "y": data['y_center'],
                "bin": bin_number
            }
            self.send_to_robot(object_data)

        # Display the image with centroids and labels
        self.display_captured_image(display_image)

        # Add summary
        summary = f"Total objects detected: {len(self.object_data)}"
        summary_label = tk.Label(self.output_frame, text=summary, bg="lightblue", fg="black", font=("Arial", 12, "bold"))
        summary_label.pack(fill="x", padx=10, pady=5)


    """
        INSPECTION-BUTTON
    """


    def inspection(self):
        if self.selected_model == "detectron2":
            self.inspect_mask()
        elif self.selected_model=="yolov8-seg":
            self.inspect_mask()
        elif self.selected_model=="custom-seg":
            self.inspect_mask()
        elif self.selected_model=="yolo-obb":
            self.get_object_dimensions()
        elif self.selected_model=="custom-obb":
            self.get_object_dimensions()
        else:
            self.terminal_frame.insert(tk.END, "No valid model selected. Hence no Inspection data\n")

    """
        INSPECTION-MASK-BUTTON
    """
    

    def inspect_mask(self):
        if not self.processed_results:
            self.terminal_frame.insert(tk.END, "No processed data available. Please capture and process an image first.\n")
            return

        if self.csv_data is None:
            self.terminal_frame.insert(tk.END, "No CSV data available. Please upload a CSV file first.\n")
            return

        self.clear_output()
        header = tk.Label(self.output_frame, text="Object   Height   Width   Status   Class Name", 
                        bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)

        for i, result in enumerate(self.processed_results):
            height = result['height']
            width = result['width']

            # Find the best match in CSV data
            best_match = None
            min_diff = float('inf')
            for _, row in self.csv_data.iterrows():
                csv_height = row['height']
                csv_width = row['width']
                diff = abs(height - csv_height) + abs(width - csv_width)
                if diff < min_diff:
                    min_diff = diff
                    best_match = row

            if best_match is not None:
                csv_height = best_match['height']
                csv_width = best_match['width']
                class_name = best_match['class name']

                height_status = self.check_tolerance(height, csv_height)
                width_status = self.check_tolerance(width, csv_width)

                if height_status == "OK" and width_status == "OK":
                    status = "OK"
                else:
                    status = "NOT OK"

                row = tk.Label(self.output_frame, 
                            text=f"Object {i+1}   {height:.2f} cm   {width:.2f} cm   {status}   {class_name}",
                            bg="gray", fg="white", font=("Arial", 10))
                row.pack(fill="x", padx=10, pady=2)

                self.terminal_frame.insert(tk.END, f"Object {i+1} (matched to {class_name}): "
                                                f"Height: {height:.2f} cm (CSV: {csv_height} cm, {height_status}), "
                                                f"Width: {width:.2f} cm (CSV: {csv_width} cm, {width_status}), "
                                                f"Status: {status}\n")
            else:
                self.terminal_frame.insert(tk.END, f"No match found for Object {i+1} in CSV data.\n")

    """
        INSPECTION-OBB-BUTTON
    """


    def get_object_dimensions(self):
        if self.csv_data is None:
            self.terminal_frame.insert(tk.END, "No CSV data available. Please upload a CSV file first.\n")
            return

        self.clear_output()

        # Add a styled header row
        header = tk.Label(self.output_frame, text="Part No  Class Name  Width   Height   CSV Width   CSV Height   Status   ", 
                        bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)

        # Create a copy of the captured image for drawing
        display_image = self.captured_image.copy()

        for part_no, data in self.object_dimensions.items():
            # Check if all required keys exist
            required_keys = ['width', 'height', 'x_center', 'y_center', 'angle']
            if not all(key in data for key in required_keys):
                missing_keys = [key for key in required_keys if key not in data]
                self.terminal_frame.insert(tk.END, f"Error: Missing data for part {part_no}. Missing keys: {', '.join(missing_keys)}. Skipping...\n")
                continue

            # Find the best match in CSV data
            best_match = None
            min_diff = float('inf')
            for _, row in self.csv_data.iterrows():
                csv_height = row['height']
                csv_width = row['width']
                diff = abs(data['height'] - csv_height) + abs(data['width'] - csv_width)
                if diff < min_diff:
                    min_diff = diff
                    best_match = row

            if best_match is not None:
                csv_width = best_match['width']
                csv_height = best_match['height']
                class_name = best_match['class name']

                width_status = self.check_tolerance(data['width'], csv_width)
                height_status = self.check_tolerance(data['height'], csv_height)

                if width_status == "OK" and height_status == "OK":
                    status = "OK"
                else:
                    status = "NOT OK"

                # Display the current object's data as a row in the frame
                row = tk.Label(self.output_frame, 
                            text=f"{part_no} {class_name}  {data['width']:.2f}   {data['height']:.2f}   {csv_width:.2f}   {csv_height:.2f}   {status}   ",
                            bg="gray" if status == "OK" else "red", fg="white", font=("Arial", 10))
                row.pack(fill="x", padx=10, pady=2)

                # Draw part number and class name on the image
                center = (int(data['x_center']), int(data['y_center']))
                cv2.putText(display_image, f"{part_no} ({class_name})", 
                            (center[0] - 20, center[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status == "OK" else (0, 0, 255), 2)

                self.terminal_frame.insert(tk.END, f"Object {part_no} (matched to {class_name}): "
                                                f"Width: {data['width']:.2f} cm (CSV: {csv_width} cm, {width_status}), "
                                                f"Height: {data['height']:.2f} cm (CSV: {csv_height} cm, {height_status}), "
                                                f"Status: {status}\n")
            else:
                self.terminal_frame.insert(tk.END, f"No match found for Object {part_no} in CSV data.\n")

        # Display the image with labels
        self.display_captured_image(display_image)

        # Add summary
        summary = f"Total objects detected: {len(self.object_data)}"
        summary_label = tk.Label(self.output_frame, text=summary, bg="lightblue", fg="black", font=("Arial", 12, "bold"))
        summary_label.pack(fill="x", padx=10, pady=5)

    """ Output Clearer"""
  
    def clear_output(self):
        """Clears all the content from the output frame."""
        for widget in self.output_frame.winfo_children():
               widget.destroy()
   


    def check_tolerance(self, measured_value, csv_value, tolerance=0.5):
        if abs(measured_value - csv_value) <= tolerance:
            return "OK"
        else:
            return "NOT OK"





if __name__ == "__main__":
    root = ctk.CTk()
    app = ObjectDetectionSoftware(root)
    root.mainloop()


                          

