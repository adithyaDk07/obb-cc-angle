  
    def process_obb(self, image):
        if self.obb_model is None:
            return "OBB model not loaded."

        results = self.obb_model(image)

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

                # Save data in dictionaries
                self.object_data[part_no] = {
                    "width": width * conversion_factor_x,
                    "height": height * conversion_factor_y,
                    "class_id": int(class_id),
                    "class_name": class_name
                }

                self.object_dimensions[part_no] = {
                    "x_center": real_x,
                    "y_center": real_y,
                    "angle": angle_degrees,
                    "class_id": int(class_id),
                    "class_name": class_name
                }

                # Draw bounding box and label on the image
                box = cv2.boxPoints(((x_center, y_center), (width, height), angle))
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                cv2.circle(image, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

                label = f"{part_no} ({class_name})"
                cv2.putText(image, label, (int(box[1][0]), int(box[1][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.display_results(image)
            return f"Total objects detected: {len(self.object_data)}"

        except Exception as e:
            return f"Error processing results: {str(e)}"




    """
        OBB-BASED
    """


    """
            SEGREGATION-OBB-BUTTON
    """
    def get_object_dimensions(self):
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

        for part_no, data in self.object_dimensions.items():
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

            # Draw bounding box, centroid, and part number on the image
            # Note: You'll need to adjust this part based on how you're storing the bounding box information
            # This is an approximation based on the center point
            center = (int(data['x_center']), int(data['y_center']))
            cv2.circle(display_image, center, 5, (255, 0, 0), -1)  # Blue circle for centroid
            cv2.putText(display_image, part_no, (center[0] - 20, center[1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the image with centroids and labels
        self.display_captured_image(display_image)

        # Add summary
        summary = f"Total objects detected: {len(self.object_)}"
        summary_label = tk.Label(self.output_frame, text=summary, bg="lightblue", fg="black", font=("Arial", 12, "bold"))
        summary_label.pack(fill="x", padx=10, pady=5)


    """
        INSPECTION-OBB-BUTTON
    """


    def get_object_data(self):
        # Clear previous content in the output frame
        for widget in self.output_frame.winfo_children():
            widget.destroy()

        # Add a styled header row
        header = tk.Label(self.output_frame, text="Part No      Width      Height      Class ID      Class Name", 
                          bg="orange", fg="black", font=("Arial", 12, "bold"))
        header.pack(fill="x", padx=10, pady=5)

        # Create a copy of the captured image for drawing
        display_image = self.captured_image.copy()

        for part_no, data in self.object_data.items():
            # Display the current object's data as a row in the frame
            row = tk.Label(self.output_frame, 
                           text=f"{part_no}      {data['width']:.2f}      {data['height']:.2f}      {data['class_id']}      {data['class_name']}",
                           bg="gray", fg="white", font=("Arial", 10))
            row.pack(fill="x", padx=10, pady=2)

            # Draw part number and class name on the image
            # Note: We'll use the center point from object_positions for drawing
            if part_no in self.object_positions:
                center = (int(self.object_positions[part_no]['x_center']), 
                          int(self.object_positions[part_no]['y_center']))
                cv2.putText(display_image, f"{part_no} ({data['class_name']})", 
                            (center[0] - 20, center[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the image with labels
        self.display_captured_image(display_image)

        # Add summary
        summary = f"Total objects detected: {len(self.object_data)}"
        summary_label = tk.Label(self.output_frame, text=summary, bg="lightblue", fg="black", font=("Arial", 12, "bold"))
        summary_label.pack(fill="x", padx=10, pady=5)

