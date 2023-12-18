import os
import argparse
import cv2
import numpy as np
'''
Adds a 1pixel-deep black padding around the entire image using OpenCV.
'''
def add_img_padding(image):
    # Define the border size (1 pixel)
    border_size = 1

    # Define the border color (black in BGR format)
    border_color = (0, 0, 0)

    # Add a 1-pixel black border to the image
    padded_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)
    
    return padded_image


'''
Filters an image using a 3x3 averaging mask in the 8-neighbourhood of each pixel.
'''
def filter_img(padded_grayscale_image, grayscale_image):
    # Get the dimensions of the grayscale image
    height, width = grayscale_image.shape[:2]

    # Create an empty matrix to store the output image with the same dimensions
    output_image = np.zeros((height, width), dtype=np.uint8)

    # Define the 3x3 filter matrix with all ones
    filter_matrix = np.array([[1/9, 1/9, 1/9],
                             [1/9, 1/9, 1/9],
                             [1/9, 1/9, 1/9]])

    # Iterate through the padded image, leaving a 1-pixel border. This will remove the padding added previously to image
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            # Extract the 3x3 neighborhood from the padded image
            neighborhood = padded_grayscale_image[y - 1:y + 2, x - 1:x + 2]

            # Perform convolution by element-wise multiplication and summation
            filtered_value = np.sum(neighborhood * filter_matrix)

            if filtered_value > 255:
                filtered_value = 255
            elif filtered_value < 0:
                filtered_value = 0

            # Assign the filtered value to the corresponding pixel in the output image
            output_image[y - 1, x - 1] = filtered_value

    return output_image

'''
Function that generates the mask image by substracting the original grayscale image - the blur image
'''
def mask_creation(original_img, blur_img):
    mask_img = np.zeros((original_img.shape[0],original_img.shape[1]),dtype=np.uint8)
    for y in range(original_img.shape[0]):
        for x in range(original_img.shape[1]):
            # Calculate the difference
            diff = original_img[y, x].astype(int) - blur_img[y, x].astype(int)
            # Clip the result to the range [0, 255]
            mask_img[y, x] = np.clip(diff[0], 0, 255).astype(np.uint8)    
    
    return mask_img


'''
Function that adds the original grayscale image to the mask image
'''
def sharpen_img(original_img, mask_img):
    sharpened_img = np.zeros((original_img.shape[0],original_img.shape[1]),dtype=np.uint8)
    for y in range(original_img.shape[0]):
        for x in range(original_img.shape[1]):
            # Calculate the sum
            summ = original_img[y, x].astype(int) + mask_img[y, x].astype(int)
            # Clip the result to the range [0, 255]
            sharpened_img[y, x] = np.clip(summ[0], 0, 255).astype(np.uint8)

    return sharpened_img

def run_unsharp(image, output_path):
    
    padded_image = add_img_padding(image)
    blurred_image = filter_img(padded_image, image)
    mask = mask_creation(image, blurred_image)
    resultant_image = sharpen_img(image, mask)
    cv2.imwrite(output_path, resultant_image)

def main():

     # Set up argparse with folder path input and modified/unmodified options
    parser = argparse.ArgumentParser(description="Script to run unsharp masking on a given folder of images and save them")
    parser.add_argument("-s", "--folder_path", action="store", help="Specify the folder path of test images", dest="folder_path", required=True)
    args = parser.parse_args()

    # Specify the test image folder path, model weights path, and output folder
    folder_path = args.folder_path
    output_image_path = os.path.abspath('../Unsharpen_Test_Images/') + '/unsharpened_{0}'
    
    # Iterate through folder and run unsharp masking
    for image in os.listdir(folder_path):
        if image.endswith(".jpg") or image.endswith(".jpeg"):
            current_image = cv2.imread(os.path.abspath(folder_path)+ '/' + image)
            # print(current_image)
            run_unsharp(current_image, output_image_path.format(image))

if __name__ == "__main__":
    main()            

