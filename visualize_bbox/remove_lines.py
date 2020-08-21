"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2
import os

INPUT_PATH = r'/home/hhn21/Documents/visualize_bbox/test'
OUT_PATH = r'/home/hhn21/Documents/visualize_bbox/result'

def no_lines_image(image):
    """
    :param image:
    :return: the image without it's lines
    """
    # Make image gray
    if len(image.shape) != 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # Blur and make binary
    blur_image = cv2.bilateralFilter(gray_image, 9, 15, 15)
    not_image = cv2.bitwise_not(blur_image)
    binary_image = cv2.adaptiveThreshold(not_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, -4)
    
    # Set the kernel size
    image_h, image_w = binary_image.shape
    vertical_size = int(image_h*2-1)
    horizontal_size = int(image_w*2-1)
    # vertical_size = int(0.9*image_h)
    # horizontal_size = int(0.9*image_w)
    horizontal, vertical = vertical_lines_image(binary_image, horizontal_size=horizontal_size, vertical_size = vertical_size)
    mark_image = vertical + horizontal
    binary_image = ~binary_image
    no_lines_image = cv2.bitwise_or(binary_image, mark_image)
    return no_lines_image

def vertical_lines_image(image, horizontal_size=35, vertical_size=21):
    # Create the images that will use to extract the horizontal and vertical lines
    vertical = np.copy(image)
    horizontal = np.copy(image)

    # Create structure element for extracting vertical lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 3))
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    # Apply morphology operations
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_size))
    vertical = cv2.dilate(vertical, verticalStructure)

    # # invert the output image
    # horizontal = 255 - horizontal
    # vertical = 255 - vertical
    return horizontal, vertical

# def remove_text_image(image, horizontal_size=50, vertical_size=50):
#     vertical = np.copy(image)
#     horizontal = np.copy(image)

#     horizontal_size = 
#     # Create structure element for extracting vertical lines through morphology operations
#     horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
#     # Apply morphology operations
#     horizontal = cv2.erode(horizontal, horizontalStructure)
#     horizontal = cv2.dilate(horizontal, horizontalStructure)

#     # Create structure element for extracting vertical lines through morphology operations
#     verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
#     # Apply morphology operations
#     vertical = cv2.erode(vertical, verticalStructure)
#     vertical = cv2.dilate(vertical, verticalStructure)

#     # # invert the output image
#     horizontal = 255 - horizontal
#     vertical = 255 - vertical
#     return horizontal, vertical

def get_list_files(input_dir):
    """
    get list name of image in input_dir
    :param input_dir: path to get list name of image
    :return: list name of image file
    """
    files = os.listdir(input_dir)
    files_image = set()
    for fn in files:
        print(fn)
        fn_lower = fn.lower()
        if fn_lower.endswith("png") or fn_lower.endswith("jpg") or fn_lower.endswith("jpeg"):
            # files_image.append(os.path.join(input_dir, fn))
            files_image.add(fn)
        elif fn_lower.endswith('pdf'):
            name = fn[:-4]
            pages = convert_from_path(os.path.join(input_dir, fn), 500)
            cnt = 0
            for page in pages:
                fp = page.save(os.path.join(input_dir, '{}_{:03d}.png'.format(name, cnt)), 'PNG')
                files_image.add('{}_{:03d}.png'.format(name, cnt))
                cnt +=1

    return files_image


def main():
    for fn in sorted(get_list_files(INPUT_PATH)):
        input_img = cv2.imread(os.path.join(INPUT_PATH, fn))
        out_img = no_lines_image(input_img)
        fname = os.path.splitext(fn)[0]
        cv2.imwrite(os.path.join(OUT_PATH, fname+"_no_lines.png"), out_img)
        # # cv2.imwrite(os.path.join(OUT_PATH, fname+"_1horizontal.png"), horizontal)
        # cv2.imwrite(os.path.join(OUT_PATH, fname+"_1vertical.png"), vertical)
        # cv2.imwrite(os.path.join(OUT_PATH, fname+"_binary.png"), binary)
        cv2.imwrite(os.path.join(OUT_PATH, fname+".png"), input_img)

if __name__ == "__main__":
    main()