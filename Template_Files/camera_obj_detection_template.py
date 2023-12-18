#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
import random
import time
import torch
import yolov5
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_segments, xyxy2xywh
from yolov5.utils.augmentations import letterbox
import pyrealsense2 as rs

# Function that initialize's the Intel RealSense D435 camera
def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    return pipeline, profile

# Function that set's up the YOLOv5 model
def setup_yolov5_model():
    device = select_device('')
    model = attempt_load('../YOLOv5_Files/best.pt', device=device) #Enter the weight name file of our model
    return model

# Function to plot a single bounding box on an image
def plot_single_bounding_box(coordinates, image, color=None, label=None, line_thickness=None):
    # Calculate the line thickness based on image size if not provided go with the alternative or
    line_thickness = line_thickness or round(0.002 * max(image.shape[0:2])) + 1
    # Generate a random color if not provided go with alternative randomizing
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    # Extract the coordinates of the bounding box
    top_left = (int(coordinates[0]), int(coordinates[1])) #top left coordinates
    bottom_right = (int(coordinates[2]), int(coordinates[3])) #bottom_right coordinates
    
    # Draw the bounding box on the image using OpenCV2
    cv2.rectangle(image, top_left, bottom_right, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    
    # If a label is provided, add it to the bounding box
    if label:
        # Calculate font thickness and size
        font_thickness = max(line_thickness - 1, 1)
        text_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        
        # Calculate the position of the label box
        label_top_left = top_left[0], top_left[1] - text_size[1] - 3
        label_bottom_right = label_top_left[0] + text_size[0], label_top_left[1] + text_size[1]
        
        # Draw a filled label box
        cv2.rectangle(image, top_left, label_bottom_right, color, -1, cv2.LINE_AA)
        
        # Add the label text to the image
        cv2.putText(image, label, (top_left[0], top_left[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

def main():
    # Initialize Intel RealSense D435 camera and YOLOv5 model using functions
    pipeline, profile = initialize_realsense()
    model = setup_yolov5_model()

    # Create a named window for displaying the object detection result using OpenCV2
    cv2.namedWindow("Object Detection using Yolov5 and Intel RealSense Camera", cv2.WINDOW_NORMAL)

    # Generate random colors for each class of the YOLOv5 Model
    random.seed(0)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(model.names))]

    while True:
        # Capture frames provided from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Address situation by continuing to the next iteration if no color frame is available
        if not color_frame:
            continue
        
        # Convert color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())
        image_original = color_image.copy()

        # Resize and preprocess the input image for YOLOv5 model
        image_resized = letterbox(image_original, new_shape=640)[0]
        image_resized = image_resized[:, :, ::-1].transpose(2, 0, 1)
        image_resized = np.ascontiguousarray(image_resized)

        # Transform the resized image to a PyTorch tensor
        image_resized = torch.from_numpy(image_resized).to(device)
        image_resized = image_resized.float() / 255.0
        image_resized = image_resized.unsqueeze(0)

        # Run the YOLOv5 model on the resized image
        predictions = model(image_resized)[0]
        predictions = non_max_suppression(predictions, 0.4, 0.5, agnostic=False)

        # Process YOLOv5 predictions and draw bounding boxes on the original image
        for i, detection in enumerate(predictions):
            if detection is not None and len(detection):
                detection[:, :4] = scale_segments(image_resized.shape[2:], detection[:, :4], image_original.shape).round()
                for *xyxy, conf, cls in reversed(detection):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_single_bounding_box(xyxy, image_original, label=label, color=colors[int(cls)], line_thickness=3)

        # Display the result in the named window using OpenCV2
        cv2.imshow("Object Detection using Yolov5 and Intel RealSense Camera D435", image_original)

        # Break the loop if the 'q' key is pressed to quit the OpenCV2 window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Stop the RealSense pipeline and close the OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()