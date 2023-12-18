import cv2
import random
import os
import argparse
import numpy as np
import torch
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_segments, xyxy2xywh
from yolov5.utils.augmentations import letterbox


# Function to plot a single bounding box on an image
def plot_single_bounding_box(coordinates, image, color=None, label=None, line_thickness=None):
    # Calculate the line thickness based on image size if not provided
    line_thickness = line_thickness or round(0.002 * max(image.shape[0:2])) + 1
    # Generate a random color if not provided
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    # Extract the coordinates of the bounding box
    top_left = (int(coordinates[0]), int(coordinates[1])) #top left coordinates
    bottom_right = (int(coordinates[2]), int(coordinates[3])) #bottom_right coordinates
    
    # Draw the bounding box on the image
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


# Function to detect objects in an image
def detect_objects(image, weights_path, output_image_path):
    # Initialize device and load YOLOv5 model
    device = select_device('')
    yolo_model = attempt_load(weights_path, device=device)

    # Define random colors for each class
    random.seed(0)
    class_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(yolo_model.names))]

    # Pre-process the input image
    input_image_letterboxed = letterbox(image, new_shape=640)[0]
    input_image_letterboxed = input_image_letterboxed[:, :, ::-1].transpose(2, 0, 1)
    input_image_letterboxed = np.ascontiguousarray(input_image_letterboxed)

     # Transform the resized image to a PyTorch tensor
    input_image_letterboxed = torch.from_numpy(input_image_letterboxed).to(device)
    input_image_letterboxed = input_image_letterboxed.float() / 255.0
    input_image_letterboxed = input_image_letterboxed.unsqueeze(0)

    # Run the YOLOv5 model on the resized image
    predictions = yolo_model(input_image_letterboxed)[0]
    predictions = non_max_suppression(predictions, 0.4, 0.5, agnostic=False)

    # Process YOLOv5 predictions and draw bounding boxes on the original image
    for i, detection in enumerate(predictions):
        if detection is not None and len(detection):
            detection[:, :4] = scale_segments(input_image_letterboxed.shape[2:], detection[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(detection):
                label = f'{yolo_model.names[int(cls)]} {conf:.2f}'
                plot_single_bounding_box(xyxy, image, label=label, color=class_colors[int(cls)], line_thickness=3)

    #Save the image to specified directory
    cv2.imwrite(output_image_path, image)   

def main():
     # Set up argparse with folder path input and modified/unmodified options
    parser = argparse.ArgumentParser(description="Script to run a specific set of YOLOV5 weights on a folder of images")
    parser.add_argument("-s", "--folder_path", action="store", help="Specify the folder path of test images", dest="folder_path", required=True)
    args = parser.parse_args()

    # Specify the test image folder path, model weights path, and output folder
    folder_path = args.folder_path
    weights_path = '../YOLOv5_Files/best.pt'
    output_image_path = os.path.abspath('../Unsharpen_Resultant_Images/') + '/yolov5_{0}'
    
    for image in os.listdir(folder_path):
        if image.endswith(".jpg") or image.endswith(".jpeg"):
            current_image = cv2.imread(os.path.abspath(folder_path)+ '\\' + image)
            detect_objects(current_image, os.path.abspath(weights_path), output_image_path.format(image))

if __name__ == "__main__":
    main()
