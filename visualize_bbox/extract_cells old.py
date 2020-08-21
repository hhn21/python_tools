import numpy as np
import sys
import cv2
import os
import ast

COLOR_TEXT = (0, 0, 255)
THICK_BOX = 2
INPUT_PATH = r'/home/hhn21/Documents/visualize_bbox/test'
OUT_PATH = r'/home/hhn21/Documents/visualize_bbox/result'
INTERSECT_THRESHOLD = 20

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
    # vertical = np.copy(image)
    horizontal = np.copy(image)
    
    # Create structure element for extracting vertical lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (image_w, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)

    # # Create structure element for extracting vertical lines through morphology operations
    # verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    # vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)

    return horizontal
#     # return horizontal + vertical

def extract_lines(image, horizontal_open_size=30, horizontal_close_size=10, vertical_open_size=20, vertical_close_size=10, joint_noise_size=10):
    vertical = np.copy(image)
    horizontal = np.copy(image)

    # Morph-close to connect dots and dash and blurred lines
    horizontal_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_close_size, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horizontal_close_kernel)
    # Morph-open with bigger kernel to extract the lines
    horizontal_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_open_size, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_open_kernel)
    # # Then we dilate to make sure the lines cross
    # horizontal = cv2.dilate(horizontal, horizontal_open_kernel)

    # Morph-close to connect dots and dash lines
    vertical_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_close_size))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vertical_close_kernel)
    # Morph-open with bigger kernel to extract the lines
    vertical_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_open_size))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, vertical_open_kernel)
    # # Then we dilate to make sure the lines cross
    # vertical = cv2.dilate(vertical, vertical_open_kernel)

    # # Joints are where horizontal lines and vertical lines meets
    # # Joints will help us to re-draw the table if it has blurred lines
    # joints = cv2.bitwise_and(horizontal, vertical)

    # # Morph-close the joints so the noise will be connected
    # # and be filtered in the finContours, because of the big size
    # joint_noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (joint_noise_size, joint_noise_size))
    # joints = cv2.morphologyEx(joints, cv2.MORPH_CLOSE, joint_noise_kernel)

    return horizontal, vertical

def combine_box(box_i, box_j):
    x_min = min(box_i[0], box_j[0])
    y_min = min(box_i[1], box_j[1])
    x_max = max(box_i[0] + box_i[2], box_j[0] + box_j[2])
    y_max = max(box_i[1] + box_i[3], box_j[1] + box_j[3])
    box = [x_min, y_min, x_max - x_min, y_max - y_min]
    return box

def combine_point(x_1, y_1, x_2, y_2):
    x_min = min(x_1, x_2)
    y_min = min(y_1, y_2)
    x_max = max(x_1, x_2)
    y_max = max(y_1, y_2)
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

    # remove the lines, if that line's width or line's height is too big
    if flag == 'horizontal':
        return [rect for rect in rects if rect[2] > 40 and rect[3] < 50 and rect[0] > 0.005*image_w and rect[0] + rect[2] < 0.995*image_w and rect[1] > 0.01*image_h and rect[1] + rect[3] < 0.995*image_h]
    elif flag == 'vertical':
        return [rect for rect in rects if rect[2] < 50 and rect[3] > 40 and rect[0] > 0.005*image_w and rect[0] + rect[2] < 0.995*image_w and rect[1] > 0.005*image_h and rect[1] + rect[3] < 0.995*image_h]


def remove_noise_boxes(boxes, image_h, image_w):
    horizontal_boxes = []
    vertical_boxes = []
    for box in boxes:
        x,y,w,h = box
        if h/w > 10:
            vertical_boxes.append(box)
        elif w/h > 10:
            horizontal_boxes.append(box)
    return horizontal_boxes, vertical_boxes

# Extend boxes until it meets other lines
# If there are no lines to meet, then create new lines in case table dont have vertical or horizontal lines
def extend_boxes(horizontal_boxes, vertical_boxes, image_h, image_w):
    buffer = 20
    horizontal_boxes = sorted(horizontal_boxes, key=lambda k: [k[1], k[0]])
    vertical_boxes = sorted(vertical_boxes, key=lambda k: [k[0], k[1]])
    extended_horizontal_boxes = []
    extended_vertical_boxes = vertical_boxes.copy()
    # Extend horizontal boxes till it meets vertical boxes at the right
    for i in range(len(horizontal_boxes)):
        box1 = horizontal_boxes[i]
        x_1, y_1, w_1, h_1 = box1
        for box3 in vertical_boxes:
            x_3, y_3, w_3, h_3 = box3
            if x_3 + w_3  + buffer > x_1+w_1 and y_3 - buffer < y_1 and y_3 + h_3 + buffer > y_1 and x_3 - x_1 - w_1< image_w*0.3:
                extended_horizontal_boxes.append([x_1, y_1, x_3+w_3-x_1,h_1])
                break
        else:
            extended_horizontal_boxes.append(box1)
            for j in range(i+1, len(horizontal_boxes)):
                box2 = horizontal_boxes[j]
                x_2, y_2, w_2, h_2 = box2
                if abs(x_2 + w_2 - x_1 - w_1) < INTERSECT_THRESHOLD and y_2 +h_2 - y_1 < image_h*0.3:
                    extended_vertical_boxes.append(combine_point(x_1 + w_1, y_1, x_2 + w_2, y_2 + h_2))
                    break
                if x_2 - buffer < x_1 and x_2 + w_2 + buffer > x_1:
                    break
    
    # Extend horizontal boxes till it meets vertical boxes at the left
    horizontal_boxes = sorted(extended_horizontal_boxes, key=lambda k: [k[1], k[0]])
    vertical_boxes = sorted(vertical_boxes, key=lambda k: [k[0], k[1]], reverse=True)
    extended_horizontal_boxes = []
    for i in range(len(horizontal_boxes)):
        box1 = horizontal_boxes[i]
        x_1, y_1, w_1, h_1 = box1
        for box3 in vertical_boxes:
            x_3, y_3, w_3, h_3 = box3
            if x_3 - 10 < x_1 and y_3 - buffer < y_1 and y_3 + h_3 + buffer > y_1 and x_1 - x_3 < image_w*0.3:
                extended_horizontal_boxes.append([x_3, y_1, x_1-x_3+w_1,h_1])
                break
        else:
            extended_horizontal_boxes.append(box1)
            for j in range(i+1, len(horizontal_boxes)):
                box2 = horizontal_boxes[j]
                x_2, y_2, w_2, h_2 = box2
                if abs(x_2 - x_1)< INTERSECT_THRESHOLD and y_2 +h_2 - y_1 < image_h*0.3:
                    extended_vertical_boxes.append(combine_point(x_1, y_1, x_2, y_2 + h_2))
                    break
                if x_2 - buffer < x_1 and x_2 + w_2 + buffer > x_1:
                    break
    

    #extend vertical boxes till it meets horizontal boxes at the top
    horizontal_boxes = sorted(extended_horizontal_boxes, key=lambda k: [k[1], k[0]], reverse=True)
    vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[0], k[1]])
    extended_vertical_boxes=[]
    for i in range(len(vertical_boxes)):
        box1 = vertical_boxes[i]
        x_1, y_1, w_1, h_1 = box1
        for box3 in horizontal_boxes:
            x_3, y_3, w_3, h_3 = box3
            if y_3 - 10 < y_1 and x_3 - buffer < x_1 and x_3 + w_3 + buffer > x_1 and y_1 - y_3 < image_h*0.3:
                extended_vertical_boxes.append([x_1, y_3, w_1, y_1 + h_1 - y_3])
                break
        else:
            extended_vertical_boxes.append(box1)
            # # Create new horizontal lines to connect the top of the vertical lines if the vertical line dont meet with any horizontal line
            # for j in range(i+1, len(vertical_boxes)):
            #     box2 = vertical_boxes[j]
            #     x_2, y_2, w_2, h_2 = box2
            #     if abs(y_2 - y_1)< INTERSECT_THRESHOLD and x_2 + w_2 - x_1 < image_w*0.3:
            #         extended_horizontal_boxes.append(combine_point(x_1, y_1, x_2 + w_2, y_2))
            #         break
            

    #extend vertical boxes till it meets horizontal boxes at the bottom
    horizontal_boxes = sorted(horizontal_boxes, key=lambda k: [k[1], k[0]])
    vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[0], k[1]])
    extended_vertical_boxes = []
    for i in range(len(vertical_boxes)):
        box1 = vertical_boxes[i]
        x_1, y_1, w_1, h_1 = box1
        for box3 in horizontal_boxes:
            x_3, y_3, w_3, h_3 = box3
            if y_3 + h_3 + buffer > y_1+h_1 and x_3 - buffer< x_1 and x_3 + w_3 + buffer > x_1 and y_3 - y_1- h_1 < image_h*0.3:
                extended_vertical_boxes.append([x_1, y_1, w_1, y_3 + h_3 - y_1])
                break
        else:
            extended_vertical_boxes.append(box1)
            # # Create new horizontal lines to connect the bottom of the vertical lines if the vertical line dont meet with any horizontal line
            # for j in range(i+1, len(vertical_boxes)):
            #     box2 = vertical_boxes[j]
            #     x_2, y_2, w_2, h_2 = box2
            #     if abs(y_2 + h_2 - y_1 - h_1)< INTERSECT_THRESHOLD  and x_2 + w_2 - x_1 < image_w*0.3:
            #         extended_horizontal_boxes.append(combine_point(x_1, y_1 + h_1, x_2 + w_2, y_2 + h_2))
            #         break
    

    horizontal_boxes = sorted(extended_horizontal_boxes, key=lambda k: [k[1], k[0]])
    vertical_boxes = sorted(extended_vertical_boxes, key=lambda k: [k[0], k[1]])
    return horizontal_boxes, vertical_boxes

# Draw lines of from the boxes
def draw_cell_lines(cell_image, horizontal_boxes, vertical_boxes):
    buffer = 5
    #Draw horizontal lines
    for box in horizontal_boxes:
        x,y,w,h = box
        start_x = x - buffer
        start_y = int(y+h/2)
        end_x = x+w + buffer
        end_y = start_y
        cell_image = cv2.line(cell_image, (start_x, start_y), (end_x, end_y), (255,255,255), 1)

    #Draw certical lines
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
    for [x, y, w, h] in list_xywh:
        binary_image[y:y+h, x:x+w] = remove_text_image(binary_image[y:y+h, x:x+w])
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_1no_text.png"), binary_image)

    # Extract the lines
    horizontal, vertical = extract_lines(binary_image)

    # create boxes of the lines
    horizontal_boxes = extract_lines_boxes(horizontal, flag='horizontal')
    vertical_boxes = extract_lines_boxes(vertical, flag='vertical')

    
    # Remove noise
    horizontal_boxes, vertical_boxes = remove_noise_boxes(horizontal_boxes+vertical_boxes,image_h,image_w)
    
    # Extend the lines till they meets
    horizontal_boxes, vertical_boxes = extend_boxes(horizontal_boxes, vertical_boxes, image_h, image_w)
    
    # Create new black image
    cell_image = np.zeros(binary_image.shape,dtype=np.uint8)
    # Draw lines
    cell_image = draw_cell_lines(cell_image, horizontal_boxes, vertical_boxes)
    
    # Debugging
    horizontal_boxes_image = draw_cells(input_image,horizontal_boxes)
    vertical_boxes_image = draw_cells(input_image,vertical_boxes)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_2horizontal.png"), horizontal)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_2vertical.png"), vertical)
    # cv2.imwrite(os.path.join(OUT_PATH, fname +"_3both.png"), horizontal+vertical)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_3horizontal_cell.png"), horizontal_boxes_image)
    cv2.imwrite(os.path.join(OUT_PATH, fname +"_3vertical_cell.png"), vertical_boxes_image)
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
    return [rect for rect in bounding_boxes if rect[2] > 20 and rect[3] >15 and rect[2]*rect[3] < image_h*image_w*0.1]




    # cv2.imwrite(os.path.join(OUT_PATH, fname +"_2horizontal.png"), horizontal)
    # cv2.imwrite(os.path.join(OUT_PATH, fname +"_2vertical.png"), vertical)
    # cv2.imwrite(os.path.join(OUT_PATH, fname +"_3both.png"), horizontal+vertical)
    # cv2.imwrite(os.path.join(OUT_PATH, fname +"_3joints.png"), joints)

    # # Draw boxes around the joints so we can connect them
    # joints_boxes = extract_joints_boxes(joints)

    # BOX_H_THRESHOLD = image_h * 0.3
    # BOX_W_THRESHOLD = image_w * 0.2
    # vertical_boxes = []
    # horizontal_boxes = []
    # joints_boxes_y = sorted(joints_boxes, key=lambda k: [k[1], k[0]])
    # joints_boxes_x = sorted(joints_boxes, key=lambda k: [k[0], k[1]])
    
    # output_image = np.zeros(binary_image.shape,dtype=np.uint8)
    # for i in range(len(joints_boxes_y)):
    #     box1 = joints_boxes_y[i]
    #     x_1, y_1, w_1, h_1 = box1
    #     for j in range(i+1, len(joints_boxes_y)):
    #         box2 = joints_boxes_y[j]
    #         x_2, y_2, w_2, h_2 = box2
    #         if abs(x_2-x_1) < INTERSECT_THRESHOLD:
    #             if abs(y_2 - y_1) > BOX_H_THRESHOLD and x_1 > image_w*0.05:
    #                 continue
    #             vertical_boxes.append(combine_box(box1, box2))
    #             output_image = cv2.line(output_image, (x_1, y_1), (x_2, y_2), (255,255,255), 2)
    #             break
    # vertical_boxes = sorted(vertical_boxes, key=lambda k: [k[0], k[1]])
    # for i in range(len(joints_boxes_x)):
    #     box1 = joints_boxes_x[i]
    #     x_1, y_1, w_1, h_1 = box1
    #     for j in range(i+1, len(joints_boxes_x)):
    #         box2 = joints_boxes_x[j]
    #         x_2, y_2, w_2, h_2 = box2
    #         if abs(y_2 - y_1) < INTERSECT_THRESHOLD/2:
    #             # horizontal_boxes.append(combine_box(box1, box2))
    #             for box3 in vertical_boxes:
    #                 x_3, y_3, w_3, h_3 = box3
    #                 if x_3 > x_1 and x_3 < x_2 and y_3 < y_1 and y_3 + h_3 > y_1:
    #                     output_image = cv2.line(output_image, (x_1, y_1), (x_3, y_1), (255,255,255), 2)
    #                     break
    #             else:
    #                 output_image = cv2.line(output_image, (x_1, y_1), (x_2, y_2), (255,255,255), 2)
    #                 break
    #             break
    
    # # get the cells
    # contours, hierarchy = cv2.findContours(output_image,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

    # # get the inner contours
    # inner_contour_indexes = []
    # if hierarchy is not None:
    #     hierarchy = hierarchy[0]
    #     for contour_idx in range(hierarchy.shape[0]):
    #         if hierarchy[contour_idx][3] != -1:
    #             inner_contour_indexes.append(contour_idx)
    
    # # get box cells
    # bounding_boxes = []
    # for contour_idx in inner_contour_indexes:
    #     bounding_rect = cv2.boundingRect(contours[contour_idx])
    #     bounding_rect = (bounding_rect[0]+1,bounding_rect[1]+1,bounding_rect[2]-2,bounding_rect[3]-2)
    #     bounding_boxes.append(bounding_rect)
    # return [rect for rect in bounding_boxes if rect[2]*rect[3] > 300], output_image

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

if __name__ == "__main__":
    for fn in sorted(get_list_files(INPUT_PATH)):
        fname = os.path.splitext(fn)[0]
        input_image = cv2.imread(os.path.join(INPUT_PATH, fn))
        if len(input_image.shape) != 2:
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = input_image
        image_h, image_w = gray_image.shape
        # blur to remove noises and make binary imagE
        blur_image = cv2.bilateralFilter(gray_image, 9, 15, 15)
        # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0blur_image.png'), blur_image)
        not_image = cv2.bitwise_not(blur_image)
        # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0not_image.png'), not_image)
        binary_image = cv2.adaptiveThreshold(not_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
        # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0binary_image7.png'), binary_image)
        
        print(fname)
        list_xywh_direction = []
        with open(os.path.join(INPUT_PATH, 'xywh_'+fname+'.txt'), 'r') as f:
            list_xywh_direction = ast.literal_eval(f.readline())
        list_xywh = [x[0] for x in list_xywh_direction]
        list_direction = [x[-1] for x in list_xywh_direction]
        
        # with open(os.path.join(INPUT_PATH, 'gt_'+fname+'.txt'), 'r') as f:
        #     lines = f.read().splitlines()
        # list_xywh = []
        # for line in lines:
        #     print(line)
        #     box = line.split(',')
        #     print(box)
        #     x= int(box[0])
        #     y= int(box[1])
        #     w= int(box[2])
        #     h= int(box[5])
        #     list_xywh.append([x,y,w,h])
        

        list_cells = extract_cell_boxes(fname,input_image,binary_image, list_xywh)
        print(len(list_cells))
        img_draw_cell = draw_cells(input_image, list_cells)

        cv2.imwrite(os.path.join(OUT_PATH, fname+'_out.png'), img_draw_cell)
        cv2.imwrite(os.path.join(OUT_PATH, fname+'.png'), input_image)

