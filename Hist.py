import cv2
import os
import math
from math import log10,sqrt
import numpy as np
import csv
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.feature import match_template

def perform_median_filter(image, kernel_size):
    median_filtered = cv2.medianBlur(image, kernel_size)
    return median_filtered    

def perform_clahe(image):
    # Check if the image is already in grayscale
    if len(image.shape) > 2 and image.shape[2] > 1:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)   

    return enhanced_image

def compute_histogram(image):
    # Check if the image is grayscale
    if len(image.shape) > 2 and image.shape[2] > 1:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute histogram
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    return histogram

def evaluate_performance(original_image, enhanced_image):
    # Check if the images are already in grayscale
    if len(original_image.shape) > 2 and original_image.shape[2] > 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if len(enhanced_image.shape) > 2 and enhanced_image.shape[2] > 1:
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    
    score = structural_similarity(original_image, enhanced_image)
    return score
def evaluate_psnr(original_image, enhanced_image):
    # Check if the images are already in grayscale
    if len(original_image.shape) > 2 and original_image.shape[2] > 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if len(enhanced_image.shape) > 2 and enhanced_image.shape[2] > 1:
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    
    mse = np.mean((original_image - enhanced_image) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Set the paths for input and output directories
input_dir = 'D:\Image_Kaggle_CLAHE\Input_image_kaggle'
output_dir = 'D:\Image_Kaggle_CLAHE\Output_image_kaggle'
input_histogram_dir = 'D:\Image_Kaggle_CLAHE\Input_Hist_kaggle'
output_histogram_dir = 'D:\Image_Kaggle_CLAHE\Output_Hist_kaggle'


# Create the output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(input_histogram_dir, exist_ok=True)
os.makedirs(output_histogram_dir, exist_ok=True)

# Create a list to store the PSNR values and ssim values
psnr_values = []
ssim_values = []


# Initialize the CSV file
csv_path = 'psnr_values.csv'
with open(csv_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Filename', 'PSNR','SSIM'])

    # Set the kernel size for median filtering
    median_kernel_size = 3

    # Loop through the images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the original image
            image_path = os.path.join(input_dir, filename)
            original_image = cv2.imread(image_path)

            # Perform median filtering
            median_filtered = perform_median_filter(original_image, median_kernel_size)

            # Perform CLAHE-based image enhancement
            enhanced_image = perform_clahe(median_filtered)

            # Save the enhanced image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, enhanced_image)

            # Compute the histogram of the input image
            input_histogram = compute_histogram(original_image)

            # Save the input histogram to the input histogram directory
            input_histogram_path = os.path.join(input_histogram_dir, f'{os.path.splitext(filename)[0]}.png')
            plt.plot(input_histogram)
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title('Input Histogram')
            plt.savefig(input_histogram_path)
            plt.close()

            # Compute the histogram of the enhanced image
            output_histogram = compute_histogram(enhanced_image)

            # Save the output histogram to the output histogram directory
            output_histogram_path = os.path.join(output_histogram_dir, f'{os.path.splitext(filename)[0]}.png')
            plt.plot(output_histogram)
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title('Output Histogram')
            plt.savefig(output_histogram_path)
            plt.close()

            print(f"{filename}: Image enhancement and histogram computation complete.")
            
            # Evaluate performance using SSIM
            
            ssim = evaluate_performance(original_image, enhanced_image)
            ssim_values.append(ssim)
            print(f"{filename}: SSIM = {ssim:.4f}")
            
            
            # Evaluate performance using PSNR
            psnr = evaluate_psnr(original_image, enhanced_image)
            psnr_values.append(psnr)
            print(f"{filename}: PSNR = {psnr:.4f}")

            # Write the filename and PSNR to the CSV file
            writer.writerow([filename, psnr,ssim])
 
# Calculate average PSNR and SSIM
average_psnr = np.mean(psnr_values) 
average_ssim = np.mean(ssim_values)

print(f"Average PSNR: {average_psnr:.4f}")
print(f"Average SSIM: {average_ssim:.4f}")

print("All images processed.")