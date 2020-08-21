import numpy as np
import cv2
import os
import json
import ast
from extract_cells import extract_cell_boxes

LONG_VERTICAL_BOX_THRESHOLD = 1.45
LONG_HORIZONTAL_BOX_THRESHOLD = 1.55

INPUT_PATH = r'/home/hhn21/Documents/visualize_bbox/test'
OUT_PATH = r'/home/hhn21/Documents/visualize_bbox/result'

# Check
def check_intersect_horiz(box1, box2, thresh=0.8):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    bb = [box1, box2]
    bb = sorted(bb, key=lambda k: k[1])
    if bb[0][1] + bb[0][3] - bb[1][1] - bb[1][3] > 0:
        intersect = bb[1][3]
    else:
        intersect = bb[0][1] + bb[0][3] - bb[1][1]
    if intersect > 0:
        # max_height = max(h1, h2)
        # if intersect / max_height > thresh:
        min_height = min(h1, h2)
        if intersect / min_height > thresh:
            return True
    return False


def check_intersect_vert(box1, box2, thresh=0.8):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    bb = [box1, box2]
    bb = sorted(bb, key=lambda k: k[0])
    if bb[0][0] + bb[0][2] - bb[1][0] - bb[1][2] > 0:
        intersect = bb[0][2]
    else:
        intersect = bb[0][0] + bb[0][2] - bb[1][0]
    if intersect > 0:
        # max_width = max(w_c, w_b)
        # if intersect / max_width > thresh:
        min_width = min(w1, w2)
        if intersect / min_width > thresh:
            return True
    return False


def add_to_dict(_dict, key, value):
    if key not in _dict:
        _dict[key] = [value]
    else:
        _dict[key].append(value)

#combine a list of boxes
def combine_box(list_box_to_merge, cell_type):
    list_xywh = [x[0] for x in list_box_to_merge]
    list_category = [x[-1] for x in list_box_to_merge]
    if cell_type == 'horizontal':
        list_xywh = sorted(list_xywh, key=lambda k: k[0])
    if cell_type == 'vertical':
        list_xywh = sorted(list_xywh, key=lambda k: k[1])
    category = 'MACHINE_PRINTED_TEXT'
    if "HANDWRITING_TEXT" in list_category:
        category = "HANDWRITING_TEXT"
    elif "BOXED_TEXT" in list_category:
        category = "BOXED_TEXT"
    elif "PARTLY_CROSS_OUT" in list_category:
        category = "PARTLY_CROSS_OUT"
    elif "MACHINE_PRINTED_TEXT" in list_category:
        if "CROSS_OUT_TEXT" in list_category:
            category = "PARTLY_CROSS_OUT"
        else:
            category = "MACHINE_PRINTED_TEXT"
    elif "NOISY_TEXT" in list_category:
        if "CROSS_OUT_TEXT" in list_category:
            category = "CROSS_OUT_TEXT"
        else:
            category = "NOISY_TEXT"
    elif "CROSS_OUT_TEXT" in list_category:
        category = "CROSS_OUT_TEXT"

    box_i = list_xywh[0]
    box_j = list_xywh[-1]
    x_min = min(box_i[0], box_j[0])
    y_min = min(box_i[1], box_j[1])
    x_max = max(box_i[0] + box_i[2], box_j[0] + box_j[2])
    y_max = max(box_i[1] + box_i[3], box_j[1] + box_j[3])
    box = [x_min, y_min, x_max - x_min, y_max - y_min]
    return (box,category)

# Get all the text box that horizontally intersect into a row
# Return a row dict of list_xywh_category
def get_rows(list_box):
    rows_dict = {}
    list_box = sorted(list_box, key=lambda k: k[0][1])
    counter = 0
    checked = []
    for i in range(len(list_box)):
        if i not in checked:
            rows_dict[counter]=[]
            box1 = list_box[i]
            rows_dict[counter].append(box1)
            for j in range(len(list_box)):
                if j != i:
                    box2 = list_box[j]
                    if check_intersect_horiz(box1[0], box2[0]):
                        rows_dict[counter].append(box2)
                        checked.append(j)
            counter += 1
    return rows_dict


#If Cell has many box in a row then it's horizontal
def check_cell_orientation(list_box):
    rows_dict = get_rows(list_box)
    for key in rows_dict:
        if len(rows_dict[key]) > 1:
            return 'horizontal'
    return 'vertical'

def has_valid_box(list_box):
    count = 0
    list_box = sorted(list_box, key=lambda k: k[0][0])
    if len(list_box) == 2:
        x,y,w,h = list_box[1][0]
        return w/h < 1.7 and h/w <1.7
    else:
        for i in range(len(list_box)):
            x,y,w,h = list_box[i][0]
            if (w/h < 1.7 or h<15) and (h/w <1.7 or w<15):
                count += 1
        return count/len(list_box) > 0.5

def has_all_valid_box(list_box):
    for box in list_box:
        x,y,w,h = box[0]
        if w/h > 1.7 or h/w >1.7:
            return False
    return True

def is_vertically_centered(list_box_to_merge, cell):
    box1 = list_box_to_merge[0]
    x1, y1, w1, h1 = box1[0]
    box2 = list_box_to_merge[-1]
    x2, y2, w2, h2 = box2[0]
    cell_x, cell_y, cell_w, cell_h = cell
    if y1 < cell_y+cell_h/2 and y2+h2 > cell_y+cell_h/2:
        return True
    return False

def has_merge_box(list_box):
    for box in list_box:
        x, y, w, h = box[0]
        if h/w > 1.7 and not w < 15:
            return True
    return False

def _check_intersect(box, cell, thresh=0.6):
    x_c, y_c, w_c, h_c = cell
    x_b, y_b, w_b, h_b = box
    if (x_c < x_b and x_b + w_b < x_c + w_c) and (y_c < y_b and y_b + h_b < y_c + h_c):
        return True
    #vertical intersect
    bb = [box, cell]
    bb = sorted(bb, key=lambda k: k[0])
    if bb[0][0] + bb[0][2] - bb[1][0] - bb[1][2] > 0:
        intersect_w = bb[0][2]
    else:
        intersect_w = bb[0][0] + bb[0][2] - bb[1][0]
    min_width = min(w_c, w_b)
    #horizontal intersect
    bb = sorted(bb, key=lambda k: k[1])
    if bb[0][1] + bb[0][3] - bb[1][1] - bb[1][3] > 0:
        intersect_h = bb[1][3]
    else:
        intersect_h = bb[0][1] + bb[0][3] - bb[1][1]
    min_height = min(h_c, h_b)
    if intersect_h / min_height > thresh and intersect_w / min_width > thresh:
        return True
    return False

def _check_box_in_cell(box, cell, thresh=0.4):
    x_c, y_c, w_c, h_c = cell
    x_b, y_b, w_b, h_b = box
    if (x_c < x_b and x_b + w_b < x_c + w_c) and (y_c < y_b and y_b + h_b < y_c + h_c):
        return True
    #vertical intersect
    bb = [box, cell]
    bb = sorted(bb, key=lambda k: k[0])
    if bb[0][0] + bb[0][2] - bb[1][0] - bb[1][2] > 0:
        intersect_w = bb[0][2]
    else:
        intersect_w = bb[0][0] + bb[0][2] - bb[1][0]
    #horizontal intersect
    bb = sorted(bb, key=lambda k: k[1])
    if bb[0][1] + bb[0][3] - bb[1][1] - bb[1][3] > 0:
        intersect_h = bb[1][3]
    else:
        intersect_h = bb[0][1] + bb[0][3] - bb[1][1]
    if intersect_h / h_b > thresh and intersect_w / w_b > thresh:
        return True
    return False

def main_process(list_xywh_category, cells, image_h, image_w):
    # make dict of boxes in cells and cells contain boxes
    cells = sorted(cells, key=lambda k: [k[1], k[0]])
    list_xywh_category = sorted(list_xywh_category, key=lambda k: [k[0][1], k[0][0]])

    boxes = [x[0] for x in list_xywh_category]
    list_category = [x[-1] for x in list_xywh_category]
    # Key: box_idx, value: list of cell_idx contain it
    box_in_cells = {}

    # These are boxes that are not in any cells or in only 1 cell
    tmp_boxes = []
    # find list of cells that contain 1 box
    # If cannot, add it to tmp_boxes
    for idx_box, box in enumerate(boxes):
        x_b, y_b, w_b, h_b = box
        is_in_cell = False
        for idx_cell, cell in enumerate(cells):
            x_c, y_c, w_c, h_c = cell
            if x_c + w_c < x_b or y_c + h_c < y_b or x_b + w_b < x_c or y_b + h_b < y_c:
                continue
            if _check_box_in_cell(box, cell):
                # add_to_dict(dict_name, key, value)
                add_to_dict(box_in_cells, idx_box, idx_cell)
                is_in_cell = True
        if not is_in_cell:
            tmp_boxes.append((boxes[idx_box], list_category[idx_box]))

    # Key: cell_idx, value: list of box_idx inside it
    cell_contain_boxes = {}
    # Set the smallest cell to be the box parent
    # If cell only contain 1 box, add it to tmp_boxes
    for idx_box, list_idx_cells in box_in_cells.items():
        if len(list_idx_cells) == 1:
            add_to_dict(cell_contain_boxes, list_idx_cells[0], idx_box)
        else:
            box_parent = [cells[i] for i in list_idx_cells]
            idx_parents = sorted(range(len(box_parent)), key=lambda k: box_parent[k][2] * box_parent[k][3])
            smallest_parent = list_idx_cells[idx_parents[0]]
            add_to_dict(cell_contain_boxes, smallest_parent, idx_box)

    final_list_xywh_category = []
    for idx_cell, list_idx_box in cell_contain_boxes.items():
        # If cell only have 1 box
        if len(list_idx_box) == 1:
            tmp_boxes.append((boxes[list_idx_box[0]],list_category[list_idx_box[0]]))
            continue
        
        list_box = []
        cell = cells[idx_cell]
        cell_x, cell_y, cell_w, cell_h = cell
        for idx_box in list_idx_box:
            list_box.append((boxes[idx_box], list_category[idx_box]))

        # Check cell type
        cell_type = check_cell_orientation(list_box)
        
        # Vertical box has to be close to the left
        # are not too long horizontally
        # and the cell has to be not to long horizontally
        # And not to close to each other 
        if cell_type == 'vertical':
            # continue
            list_box = sorted(list_box, key=lambda k: k[0][1])
            list_box_to_merge = []
            for i in range(len(list_box)-1):
                box1 = list_box[i]
                x1, y1, w1, h1 = box1[0]
                box2 = list_box[i+1]
                x2, y2, w2, h2 = box2[0]
                if y2 - y1 - h1 < 10 or not check_intersect_vert(box1[0], box2[0]):
                    break
            else:
                if len(list_box) == 2:
                    box1 = list_box[0]
                    x1, y1, w1, h1 = box1[0]
                    box2 = list_box[1]
                    x2, y2, w2, h2 = box2[0]
                    # Box close to the left
                    if x1 - cell_x < 40 and w1/h1 < 1.45 and w1 < 100 and x2 - cell_x < 40 and w2/h2 < 1.45 and w2 < 100:
                        for box in list_box:
                            list_box_to_merge.append(box)
                elif len(list_box) >5:
                    pass
                else:
                    # Boxes close to the left and right
                    box1 = list_box[0]
                    x1, y1, w1, h1 = box1[0]
                    box2 = list_box[-1]
                    x2, y2, w2, h2 = box2[0]
                    if x1 - cell_x < 50 and x2 - cell_x < 50 and w1/h1 < 1.35 and w2/h2 < 1.35 and w1 < 100 and w2 < 100 and cell_x + cell_w - x1 - w1 < 50 and cell_x + cell_w - x2 - w2 < 50 and cell_h > cell_w:
                        for box in list_box:
                            list_box_to_merge.append(box)
            
            if len(list_box_to_merge) > 0:
                if is_vertically_centered(list_box_to_merge, cell):
                    final_list_xywh_category.append(combine_box(list_box_to_merge, cell_type))
                else:
                    for box in list_box:
                        final_list_xywh_category.append(box)
            else:
                for box in list_box:
                    final_list_xywh_category.append(box)

        
        # Horizontal case
        elif cell_type == 'horizontal':
            is_big_cell = False
            rows_dict = get_rows(list_box)
            # This are for cell that has more than 2 lines
            if len(rows_dict) > 6:
                is_big_cell = True

            for key in rows_dict:
                list_box_to_merge = []
                row = rows_dict[key]
                row = sorted(row, key=lambda k: k[0][0])
                # If the row only contain 1 text box
                if len(row) == 1:
                    final_list_xywh_category.append(row[0])
                # If the row has a merge box (long vertical box)
                elif has_merge_box(row):
                    for box in row:
                        final_list_xywh_category.append(box)
                #If the row has many rows
                elif is_big_cell:
                    if len(row) > 2 and has_all_valid_box(row):
                        final_list_xywh_category.append(combine_box(row, cell_type))
                    else:
                        for box in row:
                            final_list_xywh_category.append(box)
                
                # If row only has 2 text box
                elif len(row) == 2:
                    box1 = row[0]
                    x1, y1, w1, h1 = box1[0]
                    box2 = row[-1]
                    x2, y2, w2, h2 = box2[0]
                    # Box close to both side
                    if x1 - cell_x < 90 and cell_x + cell_w - x2 - w2 < 90 and has_valid_box(row):
                        for box in row:
                            list_box_to_merge.append(box)
                    # This case has to be exclude
                    elif x1 - cell_x < 10 and cell_x + cell_w - x2 - w2 > 80:
                        final_list_xywh_category.append(box1)
                        final_list_xywh_category.append(box2)
                    # Box are close
                    elif x2 - x1 - w1 <50 and has_valid_box(row):
                        for box in row:
                            list_box_to_merge.append(box)
                    else:
                        final_list_xywh_category.append(box1)
                        final_list_xywh_category.append(box2)
                
                # If row only has 3 text box
                elif len(row) == 3:
                    box1 = row[0]
                    x1, y1, w1, h1 = box1[0]
                    box2 = row[-1]
                    x2, y2, w2, h2 = box2[0]
                    # Box are close to both sides
                    if x1 - cell_x < 90 and cell_x + cell_w - x2 - w2 < 90 and has_valid_box(row):
                        for box in row:
                            list_box_to_merge.append(box)
                    # Box are far from both sides
                    elif x1 - cell_x > 90 and cell_x + cell_w - x2 - w2 > 90 and has_valid_box(row):
                        for box in row:
                            list_box_to_merge.append(box)
                    else:
                        for box in row:
                            final_list_xywh_category.append(box)


                # If row has more than 4 text box
                else:
                    box1 = row[0]
                    x1, y1, w1, h1 = box1[0]
                    box2 = row[-1]
                    x2, y2, w2, h2 = box2[0]
                    if has_valid_box(row):
                        # Box are super close to both sides
                        if x1 - cell_x < 25 and cell_x + cell_w - x2 - w2 < 25:
                            for box in row:
                                list_box_to_merge.append(box)
                        # If the 1st and last text boxes are both close to the borders and all square
                        elif x1 - cell_x < 60 and cell_x + cell_w - x2 - w2 < 60 and (h1/w1 < 1.45 or w1 < 15) and (h2/w2 < 1.45 or w2 < 15) and (x2-x1 < 400 or (w1/h1 < 1.45 and w2/h2 < 1.45)):
                            for box in row:
                                list_box_to_merge.append(box)
                        # If the 1st and last text boxes are both far from the borders and all square
                        elif x1 - cell_x > 90 and cell_x + cell_w - x2 - w2 > 90 and (h1/w1 < 1.45 or w1 < 15) and (h2/w2 < 1.45 or w2 < 15) and (x2-x1 < 400 or (w1/h1 < 1.45 and w2/h2 < 1.45)):
                            for box in row:
                                list_box_to_merge.append(box)
                        # If the boxes are close then we merge
                        else:
                            for box in row:
                                final_list_xywh_category.append(box)

                    else:
                        for box in row:
                            final_list_xywh_category.append(box)
            
                if len(list_box_to_merge) > 0:
                    box1 = list_box_to_merge[0]
                    x1, y1, w1, h1 = box1[0]
                    box2 = list_box_to_merge[-1]
                    x2, y2, w2, h2 = box2[0]
                    if x2 - x1 > image_w * 0.6:
                        for box in row:
                            final_list_xywh_category.append(box)
                    else:
                        final_list_xywh_category.append(combine_box(list_box_to_merge, cell_type))
    
    final_list_xywh_category += tmp_boxes
    return final_list_xywh_category

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

def draw_cells(img, cells, color = (0, 180, 0), font_path='arial.pil'):
    thick_box = 2
    img_draw = img.copy()
    cells = sorted(cells, key=lambda k: [k[1], k[0]])
    for idx, ([x, y, w, h]) in enumerate(cells):
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, thick_box)
    from PIL import Image
    img_draw = Image.fromarray(img_draw)
    img_draw = np.array(img_draw)
    return img_draw

def draw_image_and_textboxes(img, list_xywh_category):
    list_xywh = [x[0] for x in list_xywh_category]
    list_category = [x[-1] for x in list_xywh_category]

    img_draw = img.copy()

    for idx, (x, y, w, h) in enumerate(list_xywh):
        # change color for hw text
        if 'HANDWRITING_TEXT' in list_category[idx]:
            COLOR_TEXT = (0, 0, 255)
        elif 'BOXED_TEXT' in list_category[idx]:
            COLOR_TEXT = (180, 180, 0)
        elif 'CROSS_OUT_TEXT' in list_category[idx]:
            COLOR_TEXT = (180, 180, 180)
        elif 'PARTLY_CROSS_OUT' in list_category[idx]:
            COLOR_TEXT = (0, 180, 180)
        elif 'BOXED_TEXT' in list_category[idx]:
            COLOR_TEXT = (180, 0, 180)
        elif 'MACHINE_PRINTED_TEXT' in list_category[idx]:
            COLOR_TEXT = (180, 0, 0)
        elif 'NOISY_TEXT' in list_category[idx]:
            COLOR_TEXT = (180, 40, 255)

        cv2.rectangle(img_draw, (x, y), (x + w, y + h), COLOR_TEXT, 3)
        cv2.putText(img_draw, str(list_category[idx][0]), (x , y), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_TEXT, 1)
    from PIL import Image, ImageDraw, ImageFont
    img_draw = Image.fromarray(img_draw)
    img_draw = np.array(img_draw)
    return img_draw

if __name__ == "__main__":
    for fn in get_list_files(INPUT_PATH):
        fname = os.path.splitext(fn)[0]
        input_image = cv2.imread(os.path.join(INPUT_PATH, fn))
        if len(input_image.shape) != 2:
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = input_image
        image_h, image_w = gray_image.shape
        # blur to remove noises and make binary image
        blur_image = cv2.bilateralFilter(gray_image, 9, 15, 15)
        # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0blur_image.png'), blur_image)
        not_image = cv2.bitwise_not(blur_image)
        # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0not_image.png'), not_image)
        binary_image = cv2.adaptiveThreshold(not_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -4)
        # cv2.imwrite(os.path.join(OUT_PATH, fname+'_0binary_image7.png'), binary_image)
        
        print(fname)
        list_xywh_category = []
        with open(os.path.join(INPUT_PATH, fname+'_xywh.txt'), 'r') as f:
            list_xywh_category = ast.literal_eval(f.readline())
        list_xywh = [x[0] for x in list_xywh_category]
        list_category = [x[-1] for x in list_xywh_category]

        list_cells = extract_cell_boxes(fname,input_image,binary_image, list_xywh)
        list_cells = list_cells if len(list_cells) > 5 else []
        
        img_draw_cell = draw_cells(input_image, list_cells)
        img_draw_cell = draw_image_and_textboxes(img_draw_cell, list_xywh_category)

        list_xywh_category = main_process(list_xywh_category, list_cells, image_h, image_w)

        
        final_list_xywh = []
        nested = False
        checked = []
        for i in range(len(list_xywh_category)):
            if i in checked:
                continue
            nested = False
            box1 = list_xywh_category[i]
            x1, y1, w1, h1 = box1[0]
            for j in range(len(list_xywh_category)):
                if j == i or j in checked:
                    continue
                box2 = list_xywh_category[j]
                x2, y2, w2, h2 = box2[0]
                if _check_intersect(box1[0], box2[0], thresh=0.8):
                    nested = True
                    if w1*h1 < w2*h2:
                        final_list_xywh.append(box2)
                        checked.append(i)
                    else:
                        final_list_xywh.append(box1)
                        checked.append(j)
                    break
            if not nested:
                final_list_xywh.append(box1)
                    
        img_xywh = draw_cells(input_image, list_cells)
        img_xywh = draw_image_and_textboxes(img_xywh, final_list_xywh)
        ####################
        # img_xywh = draw_cells(input_image, final_list_xywh, color=(0,0,255))


        cv2.imwrite(os.path.join(OUT_PATH, fname+'_xywh1.png'), img_draw_cell)
        cv2.imwrite(os.path.join(OUT_PATH, fname+'_xywh2.png'), img_xywh)
        cv2.imwrite(os.path.join(OUT_PATH, fname+'_xywh3.png'), input_image)