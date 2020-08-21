import numpy as np
import sys
import cv2
import os
import ast

COLOR_TEXT = (0, 0, 255)
THICK_BOX = 2
INPUT_PATH = r'/home/hhn21/Documents/visualize_bbox/test'
OUT_PATH = r'/home/hhn21/Documents/visualize_bbox/result'

# #remove text from the image
# # # So this method requires pixellink to do it's part correctly
# def remove_text_image(image):
#     return np.zeros(image.shape,dtype=np.uint8)

# Remove the text from the small image, keeping the lines
# So this method requires pixellink to do it's part correctly
def remove_text_image(image, horizontal_size=20, vertical_size=10):
    image_h, image_w = image.shape
    # horizontal_size = int(image_w * 0.9)
    # vertical_size = int(image_h*0.9)

    #we dont care about the lines in the middle as it's mostly text
    # and even when it's a line, it is not important, our algorithm can take care of it
    # image = cv2.rectangle(image, (0, int(image_h*0.1)), (int(image_w), int(image_h*0.9)), (0, 0, 0), cv2.FILLED)
    vertical = np.copy(image)
    horizontal = np.copy(image)
    try:
        # Create structure element for extracting vertical lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (image_w*2-1, 1))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)

        # # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image_h*2-1))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)
    except:
        pass
    # return horizontal
    return horizontal + vertical

# Extract the vertical lines and connect dashlines
# Before calling this function, it's best to heighten the box
# So that 1, M, N, L cases and so on dont get mixed up wih vertical lines 
def vertical_text_image(image, horizontal_open_size=31, horizontal_close_size=5):
    image_h, image_w = image.shape
    vertical = np.copy(image)
    horizontal = np.copy(image)
    try:
        # # Morph-close to connect dots and dash and blurred lines
        # horizontal_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_close_size, 1))
        # horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horizontal_close_kernel)
        # # Morph-open with bigger kernel to extract the lines
        # horizontal_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_open_size, 1))
        # horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_open_kernel)
        
        # Morph-open with bigger kernel to extract the lines
        horizontal_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image_w*2-1, 1))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_open_kernel)
        # # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image_h*2-1))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)
    except:
        pass
    # return vertical
    return horizontal + vertical

# Extract the horizontal and vertical lines
def horizontal_text_image(image):
    image_h, image_w = image.shape
    vertical = np.copy(image)
    horizontal = np.copy(image)
    try:
        # Morph-open with bigger kernel to extract the lines
        horizontal_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image_w*2-1, 1))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_open_kernel)

        # # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image_h*2-1))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)
    except:
        pass
    # return horizontal
    return horizontal + vertical

def extract_lines(image, horizontal_open_size=41, horizontal_close_size=11, vertical_open_size=19, vertical_close_size=11, scale=1):
    image_h, image_w = image.shape
    vertical = np.copy(image)
    horizontal = np.copy(image)
    # Morph-close to connect dots and dash and blurred lines
    horizontal_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_close_size, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horizontal_close_kernel)
    # Morph-open with bigger kernel to extract the lines
    horizontal_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_open_size, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_open_kernel)
    # Filter out noises
    horizontal_filter_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    horizontal_filter = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_filter_kernel)
    horizontal = cv2.bitwise_and(horizontal,~horizontal_filter)

    # Morph-close to connect dots and dash lines
    vertical_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_close_size))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vertical_close_kernel)
    # Morph-open with bigger kernel to extract the lines
    vertical_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_open_size))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, vertical_open_kernel)
    # Filter out noises
    vertical_filter_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vertical_filter = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, vertical_filter_kernel)
    vertical = cv2.bitwise_and(vertical,~vertical_filter)
    return horizontal, vertical

def combine_box(box_i, box_j):
    x_min = min(box_i[0], box_j[0])
    y_min = min(box_i[1], box_j[1])
    x_max = max(box_i[0] + box_i[2], box_j[0] + box_j[2])
    y_max = max(box_i[1] + box_i[3], box_j[1] + box_j[3])
    box = [x_min, y_min, x_max - x_min, y_max - y_min]
    return box

def combine_point(x1, y1, x2, y2):
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)
    w = x_max - x_min if x_max - x_min > 0 else 1
    h = y_max - y_min if y_max - y_min > 0 else 1
    box = [x_min, y_min, w, h]
    return box

def extract_joints_boxes(image):
    image_h, image_w = image.shape
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours]
    # remove the lines, if that line's width or line's height is too big
    return [rect for rect in rects if (rect[2] < 10 or rect[3] < 10) and rect[0] > 0.01*image_w and rect[0] + rect[2] < 0.99*image_w and rect[1] > 0.01*image_h and rect[1] + rect[3] < 0.99*image_h]

def extract_lines_boxes(image, flag='horizontal'):
    image_h, image_w = image.shape
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours]

    # Remove the lines
    # if that line's width or line's height is too big or the line is at the edge of the image
    if flag == 'horizontal':
        # return [rect for rect in rects if rect[2] > 40 and rect[3] < 50]
        return [rect for rect in rects if rect[2] > 40 and rect[3] < 50 and rect[0] > 0.0001*image_w and rect[0] + rect[2] < 0.9999*image_w and rect[1] > 0.0001*image_h and rect[1] + rect[3] < 0.9999*image_h]
    elif flag == 'vertical':
        # return [rect for rect in rects if rect[2] < 50 and rect[3] > 24]
        return [rect for rect in rects if rect[2] < 50 and rect[3] > 18 and rect[0] > 0.0001*image_w and rect[0] + rect[2] < 0.9999*image_w and rect[1] > 0.0001*image_h and rect[1] + rect[3] < 0.9999*image_h]


def remove_noise_boxes(boxes, image_h, image_w):
    horizontal_boxes = []
    vertical_boxes = []
    # Remove small boxes
    for box in boxes:
        x,y,w,h = box
        if h/w > 5:
            vertical_boxes.append(box)
        elif w/h > 9:
            horizontal_boxes.append(box)
    horizontal_boxes = sorted(horizontal_boxes, key=lambda k: [k[1], k[0]])
    vertical_boxes = sorted(vertical_boxes, key=lambda k: [k[0], k[1]])

    # # Remove the vertical boxes caused by characters
    # final_vertical_boxes = []
    # final_horizontal_boxes = []
    # checked = []
    # for i in range(len(vertical_boxes)):
    #     box1 = vertical_boxes[i]
    #     x1, y1, w1, h1 = box1
    #     for j in range(i+1,len(vertical_boxes)):
    #         box2 = vertical_boxes[j]
    #         x2, y2, w2, h2 = box2
    #         if x2 - x1 < 50 and x2 - x1 > 10 and abs(y1 - y2) < 20 and abs(y1 + h1 - y2 - h2) < 20 and h1 < 70 and h2 < 70:
    #             checked.append(i)
    #             checked.append(j)
    #             break
    #     else:
    #         if i not in checked:
    #             final_vertical_boxes.append(box1)

    # for i in range(len(horizontal_boxes)):
    #     box1 = horizontal_boxes[i]
    #     x1, y1, w1, h1 = box1
    #     for j in range(i+1,len(horizontal_boxes)):
    #         box2 = horizontal_boxes[j]
    #         x2, y2, w2, h2 = box2
    #         if y2 - y1 < 50 and abs(x1 - x2) < 20 and abs(x1 + w1 - x2 - w2) < 20 and w1 < 70 and w2 < 70:
    #             checked.append(i)
    #             checked.append(j)
    #             break
    #     else:
    #         if i not in checked:
    #             final_horizontal_boxes.append(box1)

    return horizontal_boxes, vertical_boxes

# Extend boxes until it meets other lines
# If there are no lines to meet, then create new lines in case table dont have vertical or horizontal lines
def extend_boxes(horizontal_boxes, vertical_boxes, image_h, image_w, scale = 1):
    INTERSECT_THRESHOLD = 15
    buffer = int(20*scale)
    connect_buffer = int(8*scale)

    
    #extend vertical boxes till it meets horizontal boxes at the top
    horizontal_boxes = sorted(horizontal_boxes, key=lambda k: [k[1], k[0]], reverse=True)
    vertical_boxes = sorted(vertical_boxes, key=lambda k: [k[0], k[1]])
    extended_vertical_boxes=[]
    for i in range(len(vertical_boxes)):
        box1 = vertical_boxes[i]
        x1, y1, w1, h1 = box1
        for box3 in horizontal_boxes:
            x3, y3, w3, h3 = box3
            if y3 - connect_buffer <= y1 and x3 - buffer < x1 and x3 + w3 + buffer > x1 and y1 - y3 < image_h*0.3:
                extended_vertical_boxes.append([x1, y3 - connect_buffer, w1, y1 + h1 + connect_buffer - y3 ])
                break
        else:
            extended_vertical_boxes.append(box1)
            # # Create new horizontal lines to connect the top of the vertical lines if the vertical line dont meet with any horizontal line
            # for j in range(i+1, len(vertical_boxes)):
            #     box2 = vertical_boxes[j]
            #     x2, y2, w2, h2 = box2
            #     if abs(y2 - y1)< INTERSECT_THRESHOLD and x2 + w2 - x1 < image_w*0.3:
            #         extended_horizontal_boxes.append(combine_point(x1, y1, x2 + w2, y2))
            #         break
    vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[0], k[1]])
    #####################################
    
    #extend vertical boxes till it meets horizontal boxes at the bottom
    horizontal_boxes = sorted(horizontal_boxes, key=lambda k: [k[1], k[0]])
    # vertical_boxes = sorted(vertical_boxes, key=lambda k: [k[0], k[1]])
    extended_vertical_boxes = []
    for i in range(len(vertical_boxes)):
        box1 = vertical_boxes[i]
        x1, y1, w1, h1 = box1
        for box3 in horizontal_boxes:
            x3, y3, w3, h3 = box3
            if y3 + h3 + connect_buffer >= y1+h1 and x3 - buffer< x1 and x3 + w3 + buffer > x1 and y3 - y1- h1 < image_h*0.3:
                extended_vertical_boxes.append([x1, y1, w1, y3 + h3 + connect_buffer - y1])
                break
        else:
            extended_vertical_boxes.append(box1)
            # # Create new horizontal lines to connect the bottom of the vertical lines if the vertical line dont meet with any horizontal line
            # for j in range(i+1, len(vertical_boxes)):
            #     box2 = vertical_boxes[j]
            #     x2, y2, w2, h2 = box2
            #     if abs(y2 + h2 - y1 - h1)< INTERSECT_THRESHOLD  and x2 + w2 - x1 < image_w*0.3:
            #         extended_horizontal_boxes.append(combine_point(x1, y1 + h1, x2 + w2, y2 + h2))
            #         break
    # vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[0], k[1]])

    # Combine vertical boxes that meets
    vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[1], k[0]])
    extended_vertical_boxes = []
    for i, box1 in enumerate(vertical_boxes):
        x1, y1, w1, h1 = box1
        for j in range(i+1, len(vertical_boxes)):
            box3 = vertical_boxes[j]
            x3, y3, w3, h3 = box3
            if abs(x3 + w3/2 - x1 - w1/2)< INTERSECT_THRESHOLD and y1 - buffer < y3 and y1 + h1 > y3:
                vertical_boxes.pop(j)
                vertical_boxes.insert(j,combine_box(box1,box3))
                break
        else:
            extended_vertical_boxes.append(box1)
    vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[0], k[1]])
    #####################################

    # Extend horizontal boxes till it meets vertical boxes at the right
    horizontal_boxes = sorted(horizontal_boxes, key=lambda k: [k[1], k[0]])
    # vertical_boxes = sorted(vertical_boxes, key=lambda k: [k[0], k[1]])
    extended_horizontal_boxes = []
    extended_vertical_boxes = vertical_boxes.copy()
    for i in range(len(horizontal_boxes)):
        box1 = horizontal_boxes[i]
        x1, y1, w1, h1 = box1
        for box3 in vertical_boxes:
            x3, y3, w3, h3 = box3
            if x3 + w3 + connect_buffer >= x1+w1 and y3 - buffer < y1 and y3 + h3 + buffer > y1 and x3 - x1 - w1< image_w*0.3:
                extended_horizontal_boxes.append([x1, y1, x3 + w3 + connect_buffer - x1,h1])
                break
        else:
            extended_horizontal_boxes.append(box1)
            # If it dont meet any vertical lines
            # It will create new vertical lines with another horizontal line that ends at the same x
            for j in range(i+1, len(horizontal_boxes)):
                box2 = horizontal_boxes[j]
                x2, y2, w2, h2 = box2
                if abs(x2 + w2 - x1 - w1) < INTERSECT_THRESHOLD and y2 +h2 - y1 < 400:
                    extended_vertical_boxes.append(combine_point(x1 + w1 - connect_buffer, y1 - connect_buffer, x2 + w2 - connect_buffer, y2 + h2 + connect_buffer))
                    break
                if x2 - buffer < x1 and x2 + w2 + buffer > x1:
                    break
    horizontal_boxes = sorted(extended_horizontal_boxes, key=lambda k: [k[1], k[0]])
    # vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[0], k[1]], reverse=True)
    #####################################

    # Extend horizontal boxes till it meets vertical boxes at the left
    # horizontal_boxes = sorted(extended_horizontal_boxes, key=lambda k: [k[1], k[0]])
    vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[0], k[1]], reverse=True)
    extended_horizontal_boxes = []
    for i in range(len(horizontal_boxes)):
        box1 = horizontal_boxes[i]
        x1, y1, w1, h1 = box1
        for box3 in vertical_boxes:
            x3, y3, w3, h3 = box3
            if x3 - connect_buffer <= x1 and y3 - buffer < y1 and y3 + h3 + buffer > y1 and x1 - x3 < image_w*0.3:
                extended_horizontal_boxes.append([x3 - connect_buffer, y1, x1-x3+w1 + connect_buffer,h1])
                break
        else:
            extended_horizontal_boxes.append(box1)
            # If it dont meet any vertical lines
            # It will create new vertical lines with another horizontal line that starts at the same x
            for j in range(i+1, len(horizontal_boxes)):
                box2 = horizontal_boxes[j]
                x2, y2, w2, h2 = box2
                if abs(x2 - x1)< INTERSECT_THRESHOLD and y2 +h2 - y1 < 400:
                    extended_vertical_boxes.append(combine_point(x1 + connect_buffer, y1 - connect_buffer, x2 + connect_buffer, y2 + h2 + connect_buffer))
                    break
                if x2 - buffer < x1 and x2 + w2 + buffer > x1:
                    break
    # horizontal_boxes = sorted(extended_horizontal_boxes, key=lambda k: [k[1], k[0]])
    vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[0], k[1]])
    
    # Combine horizontal boxes that meets
    horizontal_boxes = sorted(extended_horizontal_boxes, key=lambda k: [k[0], k[1]])
    extended_horizontal_boxes = []
    for i in range(len(horizontal_boxes)):
        box1 = horizontal_boxes[i]
        x1, y1, w1, h1 = box1
        for j in range(i+1, len(horizontal_boxes)):
            box3 = horizontal_boxes[j]
            x3, y3, w3, h3 = box3
            if abs(y3 + h3/2 - y1 - h1/2)< INTERSECT_THRESHOLD and x1 - buffer < x3  and x1 + w1 > x3:
                horizontal_boxes.pop(j)
                horizontal_boxes.insert(j,combine_box(box1,box3))
                break
        else:
            extended_horizontal_boxes.append(box1)
    horizontal_boxes = sorted(extended_horizontal_boxes, key=lambda k: [k[1], k[0]])
    # #####################################
    # horizontal_boxes = sorted(extended_horizontal_boxes, key=lambda k: [k[1], k[0]])

    return horizontal_boxes, vertical_boxes

# Draw lines of from the boxes
def draw_cell_lines(cell_image, horizontal_boxes, vertical_boxes, scale=1):
    if scale < 1:
        buffer = 0
    else:
        buffer = 6
    #Draw horizontal lines
    for box in horizontal_boxes:
        x,y,w,h = box
        start_x = x - buffer
        start_y = int(y+h/2)
        end_x = x+w + buffer
        end_y = start_y
        cell_image = cv2.line(cell_image, (start_x, start_y), (end_x, end_y), (255,255,255), 1)

    #Draw vertical lines
    for box in vertical_boxes:
        x,y,w,h = box
        start_x = int(x+w/2)
        start_y = y - buffer
        end_x = start_x
        end_y = y+h + buffer
        cell_image = cv2.line(cell_image, (start_x, start_y), (end_x, end_y), (255,255,255), 1)
    return cell_image

# Return list of cells boxes
def extract_cell_boxes(fname, input_image, binary_image, list_xywh):
    image_h, image_w = binary_image.shape
    # Remove text from the image
    scale = image_h/3507
    buffer = 5
    buffer = int(buffer * scale)

    # For every text box, we heighten it, and take out the vertical lines while connecting the dash lines
    for [x, y, w, h] in list_xywh:
        if h/w > 1.8:
            binary_image[y+buffer:y+h-buffer, x-buffer:x+w+buffer] = remove_text_image(binary_image[y+buffer:y+h-buffer, x-buffer:x+w+buffer])
        else:
            binary_image[y-buffer:y+h, x:x+w-buffer] = vertical_text_image(binary_image[y-buffer:y+h, x:x+w-buffer])
            # We then shorten the box, so that we can keep the lines near the edge
            # The result is a binary image with out texts
            binary_image[y+buffer:y+h-buffer, x+buffer:x+w-buffer] = horizontal_text_image(binary_image[y+buffer:y+h-buffer, x+buffer:x+w-buffer])


    # # Old remove text function
    # for [x, y, w, h] in list_xywh:
    #     binary_image[y:y+h-buffer, x:x+w-buffer] = remove_text_image(binary_image[y:y+h-buffer, x:x+w-buffer])

    # Extract the lines
    horizontal, vertical = extract_lines(binary_image, scale=scale)

    # create boxes of the lines
    horizontal_boxes = extract_lines_boxes(horizontal, flag='horizontal')
    vertical_boxes = extract_lines_boxes(vertical, flag='vertical')

    # # Debugging
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_1no_text.png"), binary_image)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_2horizontal.png"), horizontal)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_2vertical.png"), vertical)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_3both.png"), horizontal+vertical)
    horizontal_boxes_image = draw_cells(input_image,horizontal_boxes)
    vertical_boxes_image = draw_cells(input_image,vertical_boxes)
    both_image = draw_cells(input_image,horizontal_boxes+vertical_boxes)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_2horizontal_.png"), horizontal_boxes_image)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_2vertical_.png"), vertical_boxes_image)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_3both_.png"), both_image)
    
    # Remove noise
    horizontal_boxes, vertical_boxes = remove_noise_boxes(horizontal_boxes+vertical_boxes,image_h,image_w)
    
    # Extend the lines till they meets
    horizontal_boxes, vertical_boxes = extend_boxes(horizontal_boxes, vertical_boxes, image_h, image_w, scale)
    # Extend the lines till they meets the 2nd time
    horizontal_boxes, vertical_boxes = extend_boxes(horizontal_boxes, vertical_boxes, image_h, image_w, scale)
    
    # Create new black image
    cell_image = np.zeros(binary_image.shape,dtype=np.uint8)
    # Draw lines
    cell_image = draw_cell_lines(cell_image, horizontal_boxes, vertical_boxes, scale=scale)
    
    # # Debugging
    horizontal_boxes_image = draw_cells(input_image,horizontal_boxes)
    vertical_boxes_image = draw_cells(input_image,vertical_boxes)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_3horizontal_after.png"), horizontal_boxes_image)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_3vertical_after.png"), vertical_boxes_image)
    box_image = draw_cells(input_image.copy(), horizontal_boxes+ vertical_boxes)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_4box_image.png"), box_image)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_4cell.png"), cell_image)

    # Find the contours
    contours, hierarchy = cv2.findContours(cell_image,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    # Inner contours are the cells
    inner_contour_indexes = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for contour_idx in range(hierarchy.shape[0]):
            if hierarchy[contour_idx][3] != -1:
                inner_contour_indexes.append(contour_idx)
    # Get them box cells
    bounding_boxes = []
    for contour_idx in inner_contour_indexes:
        bounding_rect = cv2.boundingRect(contours[contour_idx])
        bounding_rect = (bounding_rect[0]+1,bounding_rect[1]+1,bounding_rect[2]-2,bounding_rect[3]-2)
        bounding_boxes.append(bounding_rect)
    return [rect for rect in bounding_boxes if rect[2] > 20 and rect[3] >15 and rect[2] < image_w*0.8 ]
    # return [rect for rect in bounding_boxes if rect[2] > 20 and rect[3] >15 and rect[2]*rect[3] < image_h*image_w*0.1]


def draw_cells(img, cells, font_path='arial.pil'):
    img_draw = img.copy()
    for idx, (x, y, w, h) in enumerate(cells):
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), COLOR_TEXT, THICK_BOX)
        cv2.putText(img_draw, str(idx), (x , y), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    COLOR_TEXT, 1)
    from PIL import Image
    img_draw = Image.fromarray(img_draw)
    img_draw = np.array(img_draw)
    return img_draw

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
    return sorted(files_image)

# if __name__ == "__main__":
#     for fn in sorted(get_list_files(INPUT_PATH)):
#         fname = os.path.splitext(fn)[0]
#         input_image = cv2.imread(os.path.join(INPUT_PATH, fn))
#         if len(input_image.shape) != 2:
#             gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray_image = input_image
#         image_h, image_w = gray_image.shape
#         # blur to remove noises and make binary image
#         blur_image = cv2.bilateralFilter(gray_image, 9, 15, 15)
#         # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0blur_image.png'), blur_image)
#         not_image = cv2.bitwise_not(blur_image)
#         # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0not_image.png'), not_image)
#         binary_image = cv2.adaptiveThreshold(not_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
#         # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0binary_image7.png'), binary_image)
        
#         print(fname)
#         list_xywh_direction = []
#         with open(os.path.join(INPUT_PATH, 'xywh_'+fname+'.txt'), 'r') as f:
#             list_xywh_direction = ast.literal_eval(f.readline())
#         list_xywh = [x[0] for x in list_xywh_direction]
#         list_direction = [x[-1] for x in list_xywh_direction]
        
#         # with open(os.path.join(INPUT_PATH, 'gt_'+fname+'.txt'), 'r') as f:
#         #     lines = f.read().splitlines()
#         # list_xywh = []
#         # for line in lines:
#         #     print(line)
#         #     box = line.split(',')
#         #     print(box)
#         #     x= int(box[0])
#         #     y= int(box[1])
#         #     w= int(box[2])
#         #     h= int(box[5])
#         #     list_xywh.append([x,y,w,h])
        

#         list_cells = extract_cell_boxes(fname,input_image,binary_image, list_xywh)
#         print(len(list_cells))
#         img_draw_cell = draw_cells(input_image, list_cells)

#         cv2.imwrite(os.path.join(OUT_PATH, fname+'_out.png'), img_draw_cell)
#         cv2.imwrite(os.path.join(OUT_PATH, fname+'.png'), input_image)

