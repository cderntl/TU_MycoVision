import os
import cv2
from tqdm import tqdm

def systematic_crop_and_save(image_path, output_folder, crop_size=640):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    base_name = os.path.basename(image_path).split('.')[0]

    crop_index = 0
    for top in range(0, height, crop_size):
        for left in range(0, width, crop_size):
            # Make sure the crop window does not go out of the image bounds
            right = min(left + crop_size, width)
            bottom = min(top + crop_size, height)

            # If the crop window is smaller than the crop size (i.e., edge case), adjust it
            if right - left < crop_size or bottom - top < crop_size:
                if right - left < crop_size:
                    left = width - crop_size
                    right = width
                if bottom - top < crop_size:
                    top = height - crop_size
                    bottom = height
            
            crop = image[top:bottom, left:right]

            output_image_path = os.path.join(output_folder, f"{base_name}_crop_{crop_index+1}.jpg")
            cv2.imwrite(output_image_path, crop)
            crop_index += 1

# Process all images in a folder
input_image_folder = "path"
output_folder = "path"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in tqdm(image_files):
    image_path = os.path.join(input_image_folder, image_file)
    systematic_crop_and_save(image_path, output_folder)