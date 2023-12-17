import os
import cv2
import math
import numpy as np


'''
Calculates the intensity matrix using equations from class
'''
def intensity_calculation(image_path):
    try:
        image = cv2.imread(image_path)

        if image is not None:
            # Get the dimensions of the image
            height, width = image.shape[:2]

            # Create an empty intensity matrix
            intensity_image = np.zeros((height, width), dtype=np.uint8)

            # Iterate through each pixel in the image
            for i in range(height):
                for j in range(width):
                    # Calculate intensity using the formula: Intensity = (Red + Green + Blue) / 3
                    intensity = np.sum(image[i, j]) // 3
                    intensity_image[i, j] = intensity

            return intensity_image

        else:
            print("\nSorry, the image could not be loaded\n")
            return None

    except Exception as e:
            print("\nError encountered while opening image using OpenCV:  "+ str(e)+'\n')
            return None

'''
Convert a set of images in a directrory to grayscale
'''
def convert_images_to_grayscale(input_directory, output_directory):

    #Get a list of all files in directory
    files = os.listdir(input_directory)

    #Initialize Iteration
    iteration = 0

    #Iterate through each file in the directory
    for  file in files:
        #Check if the file is a valid input either jpeg or jpg
        if file.endswith('.jpeg') or file.endswith('.jpg'):
            #Build full path to input image
            input_image_path = os.path.join(input_directory, file)
            #Calculate the intensity and get the grayscale image
            grayscale_image = intensity_calculation(input_image_path)
            
            if grayscale_image is not None:
                #Get the file extension
                file_extension = os.path.splitext(file)[1]

                #Build the output path
                output_image_path = os.path.join(output_directory, f"result{iteration}{file_extension}")
                
                #Save the grayscale image to the built output path
                cv2.imwrite(output_image_path, grayscale_image)
                
                #increment iteration
                iteration += 1
    
    print("Grayscale coversion completed.")



def main():
    input_directory = '../Original_Test_Images'
    output_directory = '../GrayScale_Test_Images'

    convert_images_to_grayscale(input_directory, output_directory)

if __name__ == "__main__":
    main()