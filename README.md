# YOLOv5-Image-Preprocessing-for-Vehicle-Detection
## Exploring Image Preprocessing for Improved Vehicle Detection using YOLOv5

## Purpose
The primary purpose of this project is to apply some of the image processing techniques that we were taught in this course on images that are input to a You Only Look Once (YOLO) version 5 model trained on vehicle classification and examine the effects of each process on the performance of the model. We will also be comparing our results with those obtained from inputting the unmodified images into the model. 

Specifically, this project will explore the effects of inputting the following three different versions of the same test images on the performance of the YOLOv5 model, i.e, how accurately it is able to detect and classify different vehicles: 
 - from an Unmodified RGB image that we captured around campus 
 - from the conversion of a given test image to its intensity image (as seen in the HSI sections of this course)
 - from the sharpened image using an unsharp mask (using elements of our solution to Assignment 2) upon the now-greyscaled image.

## Documents and Files Included in Repository
1. Deliverables
   - 1.1. Proposal Directory -> Project proposal and its respective evaluation by the professor
   - 1.2. Presentation Directory -> Project Slides
   - 1.3. Written Report Directory -> Project Written Report. Click the Download button as GitHub is not able to render the document due to its size.

2. Template_Files
   - 2.1. camera_obj_detection_template.py: script that uses the Intel RealSense D435 Camera and YOLOv5 weight to detect vehicles on a live feed from the camera. Bounding boxes are generated around detected objects. This file does not have established paths to required files i.e. template.
   - 2.2. single_image_object_detection_template.py: script that uses an image and the YOLOv5 weight performs object detection. The result is a saved image with the bounding boxes around objects that have been detected. This file does not have established paths to required files i.e. template.
   - 2.3. folder_images_object_detection.py: script that iterates through images from a directory, and the YOLOv5 weight performs object detection on all of the images. The result images after object detection are saved into a new directory, and the images have bounding boxes around objects that have been detected.This file does not have established paths to required files i.e. template.

The following is an example of how to run the script:
![unnamed](https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/92492748/9c707586-7e0a-47d4-a903-04f625cbee27)

3. YOLOv5_Files
   - 3.1 best.pt
        - YOLOv5 vehicle recognition weight. 
   - 3.2 Training_Notebook
        - The code in the Jupyter Notebook was created by Roboflow, and it was used to train YOLOv5 weights using the custom dataset [Vehicle Class Specification Computer Vision Project](https://universe.roboflow.com/khairul-izham-aje9q/vehicle-class-specification). More information on the dataset can be found below. The original training notebook created by Roboflow can be found here https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov5-object-detection-on-custom-data.ipynb. These resources can also be found in the Resources section of this ReadMe.md file.

4. YOLOv5_object_detection
   - There are links in the Resources section of the ReadMe.md file that aided us with the installation and learning how to utilize the libraries such as YOLOv5, PyTorch(Torch), and Pyrealsense2.
   - 4.1. camera_obj_detection_template.py: script that uses the Intel RealSense D435 camera and YOLOv5 weight to detect vehicles on a live feed from the camera. Bounding boxes are generated around detected objects. The Intel RealSense D435 camera must be set up following the links in the resources, and it must be connected to the computer work.
      - Imports Required
        ```python
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
        ``` 
   - 4.2. single_image_object_detection_template.py: script that uses an image and the YOLOv5 weight performs object detection. The result is a saved image with the bounding boxes around objects that have been detected.
      - Imports Required
        ```python
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
        ```      
   - 4.3. folder_images_object_detection.py: script that iterates through images from a directory, and the YOLOv5 weight performs object detection on all of the images. The result images after object detection are saved into a new directory, and the images have bounding boxes around objects that have been detected.
     - Imports Required
        ```python
            import cv2
            import random
            import numpy as np
            import time
            import argparse
            import os
            import torch
            import yolov5
            from yolov5.utils.torch_utils import select_device
            from yolov5.models.experimental import attempt_load
            from yolov5.utils.general import non_max_suppression, scale_segments, xyxy2xywh
            from yolov5.utils.augmentations import letterbox
        ```   
  
6. Preprocessing_scripts
   - 5.1 folder_images_grayscale_conversion.py: script that connverts all Original_Test_Images directory images to grayscale images and saves them in GrayScale_Test_Images directory. The intensity formula was used to make these calculations.
     - Imports Required
        ```python
            import os
            import cv2
            import math
            import numpy as np
        ```   

The following is an example of how to run the script:
<div align="center">
<img width="574" alt="Screenshot 2023-11-20 at 11 11 34 AM" src="https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/92492748/e3901be8-e619-49f9-9754-42cd19b08a2c">
</div>
The path to the input directory has to be modified in the directory if a different input directory is required.

   - 5.2 folder_images_unsharpening_conversion.py: script that converts all GrayScale_Test_Images directory images to unsharpen images and saves the new images in Unsharpen_Test_Images. Assignment 2 was used as a foundation for this script.
     - Imports Required
        ```python
            import os
            import argparse
            import cv2
            import numpy as np
        ```
The following is an example of how to run the script:
![unnamed (1)](https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/92492748/d50048e0-6840-4529-a57e-b88ae93f0fd9)

6. Original_Test_Images
   - This directory stores images taken by Aabir Basu and Anton Guaman that can be used for testing the YOLOv5 weight
   
7. Original_Resultant_Images
   - This directory stores the new images with bounding boxes after the script folder_images_object_detection.py was executed

8. GrayScale_Test_Images
   - This directory stores grayscale images after the preprocessing script of grayscale conversion was applied to the Original_Test_Images directory

9. GrayScale_Resultant_Images
   - This directory stores the grayscale new images with bounding boxes after the script folder_images_object_detection.py was executed
  
10. Unsharpend_Test_Images
   - This directory stores unsharpened images after the preprocessing script of unsharpening conversion was applied to the GrayScale_Test_Images directory

11. Unsharpend_Resultant_Images
   - This directory stores the unsharpened new images with bounding boxes after the script folder_images_object_detection.py was executed

## Sample Input Images
### Original RGB Colour Image

![test10](https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/97454850/0e30c83d-4a1f-4aea-9028-995b950cd0f9)

### Grayscale Image

![image (2)](https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/97454850/5132adb7-ba28-4b55-a40a-6c3f26d3a808)


### Unsharpened Image

![unsharpened_result80-min](https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/97454850/b0a74658-5511-4672-b044-7541d5df3160)

## Resultant Output Images after YOLOv5's Object Detection
### Original RGB Colour Image

![result_test10](https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/97454850/49748ff7-5f2c-4410-8791-acbba8b3d1c7)

### Grayscale Image

![image (3)](https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/97454850/8ad0fe6d-663e-4ff6-ba12-9422aff07d36)


### Unsharpened Image

![yolov5_unsharpened_result80-min](https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/97454850/8f830094-eb19-4edc-b7b3-4bf0433564df)

## Images using the Intel RealSense D435 camera

### Image of Intel RealSense Camera
![IMG_1054](https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/92492748/70f27792-3137-4c55-ba2f-6670e05e47f4)



## Vehicle_Recognition_with_Unmodified_YOLOv5.ipynb Highlights & Results
This is the YOLOv5 structure provided by Roboflow used to train our weights:
<img width="477" alt="Screenshot 2023-11-15 at 10 37 10 PM" src="https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/92492748/7ad7e26c-295d-4955-ab27-49dcb6e6d2a3">

These are the parameters that were used to train:
<img width="1253" alt="Screenshot 2023-11-15 at 10 41 22 PM" src="https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/92492748/166159f0-b148-4f19-9fb4-49a70e77a61d">

These are the resultant graphs that evaluate the custom YOLOv5 training performance
<img width="1274" alt="Screenshot 2023-11-15 at 10 42 11 PM" src="https://github.com/MUN-McIntyre/course-project-antonguaman-4155-aabirbasu-1630/assets/92492748/76a5df3c-fe2c-44b8-9128-9126ea48bf23">

### Dataset Used for Object Detection
The dataset that was the best fit for the purpose of this project was made by Khairul Izham and was found on [Roboflow Universe](https://universe.roboflow.com/). The [Vehicle Class Specification Computer Vision Project](https://universe.roboflow.com/khairul-izham-aje9q/vehicle-class-specification).

The classes that the dataset has were trained for the following vehicles:
- Bus
- Heavy 2 axies
- Heavy 3 axies  
- Light 2 axies 
- Motorcar
- Motorcycle

## Resources:
### YOLOv5 Custom Dataset Resources:
- Top Computer Vision Models Roboflow: https://roboflow.com/models
  -  Colab Notebook train-yolov5-object-detection-on-custom-data.ipynb created by Roboflow: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov5-object-detection-on-custom-data.ipynb
  -  Roboflow video guide on How to Train YOLOv5 on a Custom Dataset: https://www.youtube.com/watch?v=MdF6x6ZmLAY
  -  Roboflow article on How to Train YOLOv5 on a Custom Dataset: https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/?__hstc=169329982.2d6242a248f53b7c759099d2371f0ccd.1697052758956.1699816391852.1700490382841.6&__hssc=169329982.4.1700490382841&__hsfp=1103019690
  -  Roboflow Universe Available Datasets: https://universe.roboflow.com/
  -  Data set Used "Vehicle Class Specification Computer Vision Project": https://universe.roboflow.com/khairul-izham-aje9q/vehicle-class-specification
     -  Citation: @misc{ vehicle-class-specification_dataset,
    title = { Vehicle Class Specification Dataset },
    type = { Open Source Dataset },
    author = { Khairul Izham },
    howpublished = { \url{ https://universe.roboflow.com/khairul-izham-aje9q/vehicle-class-specification } },
    url = { https://universe.roboflow.com/khairul-izham-aje9q/vehicle-class-specification },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2023 },
    month = { oct },
    note = { visited on 2023-11-16 },
} 

- YOLOv5 Explanation of model & how to train it: https://www.exxactcorp.com/blog/Deep-Learning/YOLOv5-PyTorch-Tutorial

### Libraries Resources used in YOLOv5_Object_Detection_Scripts Folder 
#### Pyrealsense2
- Pyrealsense2 Documentation
  - Installation: https://pypi.org/project/pyrealsense2/
  - Pyrealsense2 Learn the Basics: https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html
#### PyTorch
- PyTorch Documentation
  - Installation: https://pypi.org/project/torch/
  - PyTorch Learn the Basics: https://pytorch.org/tutorials/beginner/basics/intro.html
 #### YOLOv5 
- YOLOv5 Documentation
  - Installation: https://pypi.org/project/yolov5/
  - YOLOv5 Learn Basics: https://pytorch.org/hub/ultralytics_yolov5/
  - YOLOv5 Ultralytics GitHub Repository: https://github.com/ultralytics/yolov5
    - Citation: @software{yolov5,
  title = {YOLOv5 by Ultralytics},
  author = {Glenn Jocher},
  year = {2020},
  version = {7.0},
  license = {AGPL-3.0},
  url = {https://github.com/ultralytics/yolov5},
  doi = {10.5281/zenodo.3908559},
  orcid = {0000-0001-5950-6979}
}
  
### Intel RealSense D435 Camera Setup Resources
- Intel RealSense D435 Camera Get Started: https://www.intelrealsense.com/get-started-depth-camera/

- Intel RealSense D435 Camera Set Up & Installation Documentation for Linux Operating Systems (OS): https://github.com/IntelRealSense/librealsense/blob/development/doc/distribution_linux.md

