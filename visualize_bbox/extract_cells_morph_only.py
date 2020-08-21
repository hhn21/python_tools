import numpy as np
import sys
import cv2
import os
import ast

COLOR_TEXT = (0, 0, 255)
THICK_BOX = 2
INPUT_PATH = r'/home/hhn21/Documents/visualize_bbox/test'
OUT_PATH = r'/home/hhn21/Documents/visualize_bbox/result'

# Remove the text from the small img, keeping the lines
# So this method requires pixellink to do it's part correctly
def remove_text_img(img):
    img_h, img_w = img.shape
    vertical = np.copy(img)
    horizontal = np.copy(img)
    try:
        # Create structure element for extracting vertical lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (img_w*2-1, 1))
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)

        # # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_h*2-1))
        vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)
    except:
        pass
    # return horizontal
    return horizontal + vertical

def extract_lines(img, horizontal_open_size=41, horizontal_close_size=9, vertical_open_size=19, vertical_close_size=9):
    vertical = np.copy(img)
    horizontal = np.copy(img)
    # Morph-close to connect dots and dash and blurred lines
    horizontal_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_close_size, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horizontal_close_kernel)
    # # Morph-open with bigger kernel to extract the lines
    # horizontal_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_open_size, 1))
    # horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_open_kernel)

    # Morph-close to connect dots and dash lines
    vertical_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_close_size))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vertical_close_kernel)
    # # Morph-open with bigger kernel to extract the lines
    # vertical_open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_open_size))
    # vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, vertical_open_kernel)

    return horizontal, vertical


# Return list of cells boxes
def extract_cell_boxes(fname, input_img, binary_img, list_xywh):
    img_h, img_w = binary_img.shape
    # Remove text from the img
    buffer_x = 10
    buffer_y = 10
    for [x, y, w, h] in list_xywh:
        if w < buffer_x*2:
            buffer_x = w//3
        # if h < buffer_y*2:
        #     buffer_y = h//3
        binary_img[y+buffer_y:y+h-buffer_y, x+buffer_x:x+w-buffer_x] = remove_text_img(binary_img[y+buffer_y:y+h-buffer_y, x+buffer_x:x+w-buffer_x])

    # Extract the lines
    horizontal, vertical = extract_lines(binary_img)
    
    # Draw lines
    cell_img = horizontal+vertical

    # # Debugging
    # cv2.imwrite(os.path.join(OUT_PATH, fname +"_1no_text.png"), binary_img)
    # cv2.imwrite(os.path.join(OUT_PATH, fname +"_2horizontal.png"), horizontal)
    # cv2.imwrite(os.path.join(OUT_PATH, fname +"_2vertical.png"), vertical)
    # cv2.imwrite(os.path.join(OUT_PATH, fname +"_3both.png"), cell_img)

    # Find the contours
    contours, hierarchy = cv2.findContours(cell_img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
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
    return [rect for rect in bounding_boxes if rect[2]*rect[3]>400 and rect[2] < img_w*0.8 ]
    # return [rect for rect in bounding_boxes if rect[2] > 20 and rect[3] >15 and rect[2] < img_w*0.8 ]
    # return [rect for rect in bounding_boxes]
    # return [rect for rect in bounding_boxes if rect[2] > 20 and rect[3] >15 and rect[2]*rect[3] < img_h*img_w*0.1]


def draw_cells(img, cells, font_path='arial.pil'):
    img_draw = img.copy()
    for idx, (x, y, w, h) in enumerate(cells):
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), COLOR_TEXT, THICK_BOX)
        cv2.putText(img_draw, str(idx), (x , y), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    COLOR_TEXT, 1)
    from PIL import img
    img_draw = img.fromarray(img_draw)
    img_draw = np.array(img_draw)
    return img_draw

def get_list_files(input_dir):
    """
    get list name of img in input_dir
    :param input_dir: path to get list name of img
    :return: list name of img file
    """
    files = os.listdir(input_dir)
    files_img = set()
    for fn in files:
        print(fn)
        fn_lower = fn.lower()
        if fn_lower.endswith("png") or fn_lower.endswith("jpg") or fn_lower.endswith("jpeg"):
            # files_img.append(os.path.join(input_dir, fn))
            files_img.add(fn)
    return sorted(files_img)

# if __name__ == "__main__":
#     for fn in sorted(get_list_files(INPUT_PATH)):
#         fname = os.path.splitext(fn)[0]
#         input_img = cv2.imread(os.path.join(INPUT_PATH, fn))
#         if len(input_img.shape) != 2:
#             gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
#         else:
#             gray_img = input_img
#         img_h, img_w = gray_img.shape
#         # blur to remove noises and make binary img
#         blur_img = cv2.bilateralFilter(gray_img, 9, 15, 15)
#         # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0blur_img.png'), blur_img)
#         not_img = cv2.bitwise_not(blur_img)
#         # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0not_img.png'), not_img)
#         binary_img = cv2.adaptiveThreshold(not_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
#         # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0binary_img7.png'), binary_img)
        
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
        

#         list_cells = extract_cell_boxes(fname,input_img,binary_img, list_xywh)
#         print(len(list_cells))
#         img_draw_cell = draw_cells(input_img, list_cells)

#         cv2.imwrite(os.path.join(OUT_PATH, fname+'_out.png'), img_draw_cell)
#         cv2.imwrite(os.path.join(OUT_PATH, fname+'.png'), input_img)

