import numpy as np
import sys
import cv2
import os

COLOR_TEXT = (0, 0, 255)
THICK_BOX = 2
INPUT_PATH = r'/home/hhn21/Documents/visualize_bbox/test'
OUT_PATH = r'/home/hhn21/Documents/visualize_bbox/result'

class ExtractCells(object):
    def __init__(self, image, list_xywh_direction, out_dir_debug=None, out_fn_debug=None):
        if len(image.shape) != 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        image_h, image_w = gray_image.shape

        # blur to remove noises and make binary imag
        blur_image = cv2.bilateralFilter(gray_image, 9, 15, 15)
        not_image = cv2.bitwise_not(blur_image)
        binary_image = cv2.adaptiveThreshold(not_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -7)

        self.image_h = image_h
        self.image_w = image_w
        self.input_image = image
        self.gray_image = gray_image
        self.binary_image = binary_image

        list_xywh = [x[0] for x in list_xywh_direction]
        list_direction = [x[-1] for x in list_xywh_direction]
        self.list_xywh = list_xywh
        self.list_direction = list_direction

        self.out_dir_debug = out_dir_debug
        self.out_fn_debug = out_fn_debug

    def get_lines_image(self, horizontal_size=10, vertical_size=10):
        # Create the images that will be used to extract the horizontal and vertical lines
        horizontal = np.copy(self.binary_image)
        vertical = np.copy(self.binary_image)

        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        # Show extracted vertical lines
        # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 200))
        # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 2))
        # horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, horizontal_kernel)
        # vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, vertical_kernel)

        # fill_rects(horizontal)
        # fill_rects(vertical)

        # invert the output image
        horizontal = 255 - horizontal
        vertical = 255 - vertical
        # cv2.imwrite(os.path.join(OUT_PATH, "vertical.png"), vertical)
        # cv2.imwrite(os.path.join(OUT_PATH, "horizontal.png"), horizontal)
        return horizontal, vertical

    def extract_line_rect(self, image, remove_line_at_border=True):
        """
        :param image: The image contains horizontal lines or vertical lines.
        :param remove_line_at_border: True: remove the line reach to the border of image. False: Otherwise
        :return: Extract all line in the image
        """
        extract_line_image = image.copy()
        image_shape = image.shape
        PADDING_SIZE = 5

        # add padding to the image to extract line reach to border image
        extract_line_image = Utils.expand_image(extract_line_image, PADDING_SIZE)

        contours = cv2.findContours(extract_line_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        rects = [cv2.boundingRect(c) for c in contours]

        # Remove noise: small line or the line reach to border image(It depends on the remove_line_at_border parameter)
        # convert box coordination from padding image to the original image
        rects = [Utils.remove_padding_rect(rect, image_shape, PADDING_SIZE) for rect in rects if
                 not (Utils.is_noise(rect))
                 and not (remove_line_at_border and Utils.line_at_border_image(rect, image_shape, PADDING_SIZE))]
        # remove the lines, if that line's width or line's height is too big
        return [rect for rect in rects if rect[2] < 40 or rect[3] < 40]

    def extract_line(self):
        """
        :param image:
        :return: extract all of line from the image
        """
        # extract horizontal, vertical lines
        horizontal, vertical = self.get_lines_image()

        # extract bounding box of these horizontal lines in the image from horizontal line image
        h_rects = self.extract_line_rect(horizontal)
        # extract bounding box of these vertical lines in the image from vertical line image
        v_rects = self.extract_line_rect(vertical)

        # remove noise "vertical" line rects if it get from reverse area #12828
        v_rects = Utils.remove_reverse_line_rects(v_rects, self.binary_image, orientation="v")

        # # remove duplicated bounding rectangle of these horizontal lines
        # h_rects = Utils.remove_duplicate_line_rects(h_rects, "horizontal")
        # # remove duplicated bounding rectangle of these vertical lines
        # v_rects = Utils.remove_duplicate_line_rects(v_rects, "vertical")

        line_rects = h_rects + v_rects

        # find text blocks
        text_block_rects = self.text_blocks.copy()

        # extract the image without it's lines from the original image
        no_line_image = ImageProcessing.no_lines_image(image)

        # remove all text blocks in the no lines image
        new_no_line_image = Utils.fill_boxes(no_line_image.copy(), text_block_rects, (255, 255, 255))

        # detect the rest of the text in the image. Because the text blocks are detected by Infordio pixellink is not enough

        # remove line, which insides text blocks
        invalid_line_rects = []

        for text_block_rect in text_block_rects:
            for line_rect in line_rects:
                # Do not check the long line
                # if line_rect[2] > 180 or line_rect[3] > 100:
                #     continue
                # remove small line insides the text blocks, shape or logo, etc.
                if Utils.line_inside_text_block(text_block_rect, line_rect, 12) and Utils.real_text_block(text_block_rect):
                    invalid_line_rects.append(line_rect)

        # remove the invalid lines
        for line in invalid_line_rects:
            if line in line_rects:
                line_rects.remove(line)

        # find the position of the document in the image
        document_rect = ImageProcessing.document_detection(self.input_image.copy())
        # keep the lines which insides the document area
        line_rects = [line_rect for line_rect in line_rects if Utils.rect_inside_rect(document_rect, line_rect)]

        # remove the small and independence line
        line_rects = Utils.remove_noise_lines_rect(line_rects, text_block_rects)

        lines = []

        # convert bounding rectangle line to Line object
        for box in line_rects:
            lines.append(Utils.convert_box_to_line(box))

        # remove underline text if a line is center alignment with the text block
        lines, self.text_blocks = Utils.remove_underline_text(lines, self.text_blocks.copy())

        # extract the line, which is made by dots
        dot_rects = self.extract_dot_lines(image.copy(), no_line_image.copy(), self.text_blocks.copy())
        # add dotted lines to process like a normal line
        for box in dot_rects:
            lines.append(Utils.convert_box_to_line(box))

        # divide those line into 2 list of lines: horizontal lines and vertical lines
        h_lines, v_lines = Utils.split_lines(lines)

        # The horizontal line and vertical line is reduced if it's intersection is inside the line and
        # distance between start point (end point) and this intersection is smaller than threshold
        h_lines, v_lines = Utils.reduce_line(h_lines, v_lines, 18) #14560(15->18)

        # The horizontal line and vertical line is extended to each others when the distance between
        # start point (end point) of horizontal (vertical) and start point (end point) of vertical (horizontal)
        # is smaller than threshold
        h_lines, v_lines = Utils.extend_line_at_corner(h_lines, v_lines, 21, 21) #14560(15->21)

        # remove small invalid line
        h_lines = Utils.remove_invalid_signed(h_lines)

        # remove overlap text block lines (#12585)
        # h_lines = self.remove_lines_overlap_text_blocks(h_lines, self.text_blocks, orientation="h")
        v_lines = self.remove_lines_overlap_text_blocks(v_lines, self.text_blocks, orientation="v")

        lines = h_lines + v_lines

        # reset connect_at_corner attribute of a line for processing in the rear path
        lines = Utils.reset_connect_at_line_corner(lines)

        # create the image that contains horizontal lines and vertical lines, which is extracted before
        lines_image = horizontal.copy()
        lines_image[vertical == 0] = 0

        return lines, lines_image, no_line_image

    def extract_box_cells(self):
        # extract list of lines object(coordination)
        # image contains horizontal lines and vertical lines
        # image do not contain any lines
        lines, lines_image, no_line_image = self.extract_line()

        # extract the bounding rectangle of these tables inside the image
        boxes = self.extract_table_boxes(lines)

        # remove the small boxes and all the lines inside those boxes
        boxes, lines = self.refine_boxes_and_lines(boxes, lines)

        # Find all possible text blocks (as content blocks)
        content_blocks = self.content_blocks()

        table_processing = TableProcessing(self.input_image, self.binary_image, self.crnn_chars, self.header_regex, self.flag_semi_virtual, self.debug_save_path)

        table_boxes = [TableBox(box, True) for box in boxes]
        tables = table_processing.process(table_boxes, lines, content_blocks, self.use_virtual_lines_for_blocks)

        boxes = self.extract_boxes(gray_image.shape, tables)

        lines_processed_image = np.zeros(self.input_image.shape,dtype=np.uint8)
        lines_processed_image.fill(0) # or img[:] = 255

        # this will draw all the lines (frame, real, maybe virtual) for each table object
        for idx, table in enumerate(tables):
            Utils.draw_table(lines_processed_image, table, draw_virtual_lines = self.use_virtual_lines_for_blocks)
        # cv2.imwrite('/home/vsocr/hanh/test_results/debug/lines_processed_image.png', lines_processed_image)
        #binarize
        lines_gray_image = cv2.cvtColor(lines_processed_image, cv2.COLOR_BGR2GRAY)

        ret, lines_bin_image = cv2.threshold(lines_gray_image,1,255,cv2.THRESH_BINARY)


        # get the cells
        image, contours, hierarchy = cv2.findContours(lines_bin_image,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

        # get the inner contours
        inner_contour_indexes = []
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for contour_idx in range(hierarchy.shape[0]):
                if hierarchy[contour_idx][3] != -1:
                    inner_contour_indexes.append(contour_idx)
        
        # get box cells
        bounding_boxes = []
        for contour_idx in inner_contour_indexes:
            bounding_rect = cv2.boundingRect(contours[contour_idx])
            bounding_rect = (bounding_rect[0]+1,bounding_rect[1]+1,bounding_rect[2]-2,bounding_rect[3]-2)
            if not Utils.is_rect_divide_content(bounding_rect, content_blocks): # ignore rect if divide content #13993
                bounding_boxes.append(bounding_rect)
        return bounding_boxes

def fill_rects(image, color=(255, 255, 255)):
    """Fill all rects in the input image with the color
    """
    contours = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    rects = [cv2.boundingRect(c) for c in contours]
    rects = [rect for rect in rects if rect[2] < 10 or rect[3] < 10]
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), color, cv2.FILLED)
    return image


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

def main():
    for fn in sorted(get_list_files(INPUT_PATH)):
        input_img = cv2.imread(os.path.join(INPUT_PATH, fn))
        out_img = lines_image(input_img)
        fname = os.path.splitext(fn)[0]
        cv2.imwrite(os.path.join(OUT_PATH, fname+"_lines.png"), out_img)
        cv2.imwrite(os.path.join(OUT_PATH, fname+".png"), input_img)

if __name__ == "__main__":
    list_xywh_direction = [([3872, 79, 324, 45], 'MACHINE_PRINTED_TEXT'), ([4208, 79, 267, 45], 'MACHINE_PRINTED_TEXT'), ([4061, 136, 140, 46], 'MACHINE_PRINTED_TEXT'), ([4205, 136, 276, 50], 'MACHINE_PRINTED_TEXT'), ([1734, 132, 551, 64], 'MACHINE_PRINTED_TEXT'), ([2323, 144, 1265, 101], 'MACHINE_PRINTED_TEXT'), ([3944, 193, 380, 56], 'MACHINE_PRINTED_TEXT'), ([1896, 200, 67, 64], 'MACHINE_PRINTED_TEXT'), ([2059, 200, 71, 60], 'MACHINE_PRINTED_TEXT'), ([2221, 200, 67, 60], 'MACHINE_PRINTED_TEXT'), ([3638, 200, 271, 49], 'MACHINE_PRINTED_TEXT'), ([3955, 303, 102, 45], 'MACHINE_PRINTED_TEXT'), ([4069, 318, 452, 56], 'MACHINE_PRINTED_TEXT'), ([1885, 302, 146, 52], 'MACHINE_PRINTED_TEXT'), ([2504, 310, 56, 214], 'MACHINE_PRINTED_TEXT'), ([2032, 317, 423, 64], 'MACHINE_PRINTED_TEXT'), ([3951, 355, 109, 48], 'MACHINE_PRINTED_TEXT'), ([1885, 363, 42, 38], 'MACHINE_PRINTED_TEXT'), ([1938, 363, 87, 42], 'MACHINE_PRINTED_TEXT'), ([3951, 427, 109, 48], 'MACHINE_PRINTED_TEXT'), ([1889, 431, 136, 49], 'MACHINE_PRINTED_TEXT'), ([2037, 450, 71, 57], 'MACHINE_PRINTED_TEXT'), ([4254, 454, 94, 49], 'MACHINE_PRINTED_TEXT'), ([4356, 454, 106, 51], 'MACHINE_PRINTED_TEXT'), ([4477, 454, 45, 49], 'MACHINE_PRINTED_TEXT'), ([1885, 484, 143, 52], 'MACHINE_PRINTED_TEXT'), ([3951, 484, 109, 48], 'MACHINE_PRINTED_TEXT'), ([2467, 586, 90, 52], 'MACHINE_PRINTED_TEXT'), ([249, 593, 90, 52], 'MACHINE_PRINTED_TEXT'), ([4133, 612, 68, 50], 'MACHINE_PRINTED_TEXT'), ([4205, 612, 79, 50], 'MACHINE_PRINTED_TEXT'), ([4303, 612, 147, 50], 'MACHINE_PRINTED_TEXT'), ([1466, 616, 140, 49], 'MACHINE_PRINTED_TEXT'), ([1625, 616, 147, 49], 'MACHINE_PRINTED_TEXT'), ([1916, 616, 230, 49], 'MACHINE_PRINTED_TEXT'), ([2180, 616, 52, 49], 'MACHINE_PRINTED_TEXT'), ([2641, 616, 238, 46], 'MACHINE_PRINTED_TEXT'), ([2894, 616, 60, 46], 'MACHINE_PRINTED_TEXT'), ([3101, 612, 52, 56], 'MACHINE_PRINTED_TEXT'), ([3290, 612, 48, 56], 'MACHINE_PRINTED_TEXT'), ([3479, 612, 52, 52], 'MACHINE_PRINTED_TEXT'), ([3683, 616, 136, 46], 'MACHINE_PRINTED_TEXT'), ([3846, 616, 60, 42], 'MACHINE_PRINTED_TEXT'), ([3925, 616, 67, 42], 'MACHINE_PRINTED_TEXT'), ([423, 616, 317, 56], 'MACHINE_PRINTED_TEXT'), ([888, 616, 48, 52], 'MACHINE_PRINTED_TEXT'), ([1258, 616, 60, 56], 'MACHINE_PRINTED_TEXT'), ([1073, 619, 48, 49], 'MACHINE_PRINTED_TEXT'), ([2467, 646, 90, 48], 'MACHINE_PRINTED_TEXT'), ([249, 650, 94, 48], 'MACHINE_PRINTED_TEXT'), ([3211, 703, 267, 59], 'MACHINE_PRINTED_TEXT'), ([1575, 706, 188, 56], 'MACHINE_PRINTED_TEXT'), ([3898, 706, 83, 56], 'MACHINE_PRINTED_TEXT'), ([2591, 710, 272, 48], 'MACHINE_PRINTED_TEXT'), ([2897, 710, 117, 52], 'MACHINE_PRINTED_TEXT'), ([3604, 710, 218, 48], 'MACHINE_PRINTED_TEXT'), ([370, 714, 49, 56], 'MACHINE_PRINTED_TEXT'), ([597, 714, 48, 52], 'MACHINE_PRINTED_TEXT'), ([748, 725, 56, 41], 'MACHINE_PRINTED_TEXT'), ([2388, 764, 56, 60], 'MACHINE_PRINTED_TEXT'), ([2388, 832, 56, 60], 'MACHINE_PRINTED_TEXT'), ([1609, 767, 154, 56], 'MACHINE_PRINTED_TEXT'), ([2493, 767, 48, 63], 'MACHINE_PRINTED_TEXT'), ([2728, 779, 124, 37], 'MACHINE_PRINTED_TEXT'), ([2588, 782, 53, 38], 'MACHINE_PRINTED_TEXT'), ([3604, 774, 124, 49], 'MACHINE_PRINTED_TEXT'), ([370, 778, 52, 56], 'MACHINE_PRINTED_TEXT'), ([597, 778, 52, 48], 'MACHINE_PRINTED_TEXT'), ([2943, 778, 37, 41], 'MACHINE_PRINTED_TEXT'), ([748, 786, 56, 52], 'MACHINE_PRINTED_TEXT'), ([1609, 831, 154, 56], 'MACHINE_PRINTED_TEXT'), ([2580, 831, 279, 56], 'MACHINE_PRINTED_TEXT'), ([3600, 839, 131, 48], 'MACHINE_PRINTED_TEXT'), ([374, 847, 50, 45], 'MACHINE_PRINTED_TEXT'), ([431, 847, 212, 45], 'MACHINE_PRINTED_TEXT'), ([2939, 842, 44, 37], 'MACHINE_PRINTED_TEXT'), ([748, 846, 56, 48], 'MACHINE_PRINTED_TEXT'), ([3592, 891, 124, 60], 'MACHINE_PRINTED_TEXT'), ([934, 900, 366, 49], 'MACHINE_PRINTED_TEXT'), ([1605, 895, 158, 56], 'MACHINE_PRINTED_TEXT'), ([2387, 895, 60, 64], 'MACHINE_PRINTED_TEXT'), ([2580, 895, 60, 56], 'MACHINE_PRINTED_TEXT'), ([3827, 895, 150, 60], 'MACHINE_PRINTED_TEXT'), ([2810, 899, 53, 52], 'MACHINE_PRINTED_TEXT'), ([3139, 899, 147, 56], 'MACHINE_PRINTED_TEXT'), ([374, 907, 45, 48], 'MACHINE_PRINTED_TEXT'), ([559, 907, 75, 48], 'MACHINE_PRINTED_TEXT'), ([480, 910, 56, 45], 'MACHINE_PRINTED_TEXT'), ([2493, 948, 45, 67], 'MACHINE_PRINTED_TEXT'), ([1605, 956, 158, 56], 'MACHINE_PRINTED_TEXT'), ([3173, 956, 309, 59], 'MACHINE_PRINTED_TEXT'), ([3857, 956, 124, 59], 'MACHINE_PRINTED_TEXT'), ([1069, 959, 233, 56], 'MACHINE_PRINTED_TEXT'), ([2580, 959, 60, 60], 'MACHINE_PRINTED_TEXT'), ([2807, 963, 59, 52], 'MACHINE_PRINTED_TEXT'), ([2897, 963, 117, 52], 'MACHINE_PRINTED_TEXT'), ([374, 971, 201, 49], 'MACHINE_PRINTED_TEXT'), ([578, 975, 65, 42], 'MACHINE_PRINTED_TEXT'), ([748, 978, 56, 41], 'MACHINE_PRINTED_TEXT'), ([3600, 1016, 116, 60], 'MACHINE_PRINTED_TEXT'), ([1386, 1020, 336, 60], 'MACHINE_PRINTED_TEXT'), ([2584, 1024, 59, 56], 'MACHINE_PRINTED_TEXT'), ([2807, 1024, 52, 56], 'MACHINE_PRINTED_TEXT'), ([3139, 1024, 147, 59], 'MACHINE_PRINTED_TEXT'), ([3827, 1024, 146, 59], 'MACHINE_PRINTED_TEXT'), ([2693, 1027, 60, 53], 'MACHINE_PRINTED_TEXT'), ([366, 1031, 56, 52], 'MACHINE_PRINTED_TEXT'), ([559, 1031, 45, 52], 'MACHINE_PRINTED_TEXT'), ([748, 1031, 48, 52], 'MACHINE_PRINTED_TEXT'), ([2387, 1035, 60, 354], 'MACHINE_PRINTED_TEXT'), ([2694, 1092, 56, 49], 'MACHINE_PRINTED_TEXT'), ([2584, 1096, 53, 45], 'MACHINE_PRINTED_TEXT'), ([2811, 1096, 45, 45], 'MACHINE_PRINTED_TEXT'), ([3139, 1088, 147, 60], 'MACHINE_PRINTED_TEXT'), ([3823, 1092, 150, 52], 'MACHINE_PRINTED_TEXT'), ([170, 1107, 63, 59], 'MACHINE_PRINTED_TEXT'), ([2493, 1145, 41, 67], 'MACHINE_PRINTED_TEXT'), ([3305, 1145, 215, 59], 'MACHINE_PRINTED_TEXT'), ([3823, 1148, 154, 56], 'MACHINE_PRINTED_TEXT'), ([1409, 1156, 269, 49], 'MACHINE_PRINTED_TEXT'), ([1712, 1156, 90, 49], 'MACHINE_PRINTED_TEXT'), ([1866, 1156, 169, 48], 'MACHINE_PRINTED_TEXT'), ([2089, 1156, 173, 48], 'MACHINE_PRINTED_TEXT'), ([2584, 1156, 267, 48], 'MACHINE_PRINTED_TEXT'), ([446, 1160, 48, 52], 'MACHINE_PRINTED_TEXT'), ([744, 1160, 48, 52], 'MACHINE_PRINTED_TEXT'), ([366, 1167, 53, 158], 'MACHINE_PRINTED_TEXT'), ([1409, 1217, 269, 53], 'MACHINE_PRINTED_TEXT'), ([1681, 1224, 121, 42], 'MACHINE_PRINTED_TEXT'), ([2656, 1216, 173, 52], 'MACHINE_PRINTED_TEXT'), ([2844, 1216, 173, 56], 'MACHINE_PRINTED_TEXT'), ([446, 1220, 48, 56], 'MACHINE_PRINTED_TEXT'), ([748, 1220, 48, 56], 'MACHINE_PRINTED_TEXT'), ([1866, 1220, 169, 52], 'MACHINE_PRINTED_TEXT'), ([2093, 1220, 165, 48], 'MACHINE_PRINTED_TEXT'), ([2580, 1235, 56, 94], 'MACHINE_PRINTED_TEXT'), ([1598, 1281, 76, 45], 'MACHINE_PRINTED_TEXT'), ([1409, 1289, 170, 45], 'MACHINE_PRINTED_TEXT'), ([1674, 1289, 131, 41], 'MACHINE_PRINTED_TEXT'), ([748, 1281, 48, 59], 'MACHINE_PRINTED_TEXT'), ([1866, 1281, 169, 55], 'MACHINE_PRINTED_TEXT'), ([2089, 1281, 173, 52], 'MACHINE_PRINTED_TEXT'), ([2663, 1281, 162, 55], 'MACHINE_PRINTED_TEXT'), ([2848, 1281, 169, 55], 'MACHINE_PRINTED_TEXT'), ([446, 1288, 52, 52], 'MACHINE_PRINTED_TEXT'), ([2580, 1330, 60, 59], 'MACHINE_PRINTED_TEXT'), ([1409, 1349, 178, 49], 'MACHINE_PRINTED_TEXT'), ([1598, 1349, 208, 49], 'MACHINE_PRINTED_TEXT'), ([2652, 1341, 59, 56], 'MACHINE_PRINTED_TEXT'), ([748, 1345, 48, 56], 'MACHINE_PRINTED_TEXT'), ([2773, 1345, 56, 52], 'MACHINE_PRINTED_TEXT'), ([370, 1349, 52, 52], 'MACHINE_PRINTED_TEXT'), ([555, 1349, 56, 48], 'MACHINE_PRINTED_TEXT'), ([1866, 1349, 181, 48], 'MACHINE_PRINTED_TEXT'), ([2089, 1349, 173, 48], 'MACHINE_PRINTED_TEXT'), ([280, 1401, 44, 64], 'MACHINE_PRINTED_TEXT'), ([3135, 1405, 347, 56], 'MACHINE_PRINTED_TEXT'), ([3823, 1405, 154, 56], 'MACHINE_PRINTED_TEXT'), ([174, 1409, 59, 60], 'MACHINE_PRINTED_TEXT'), ([2580, 1409, 317, 52], 'MACHINE_PRINTED_TEXT'), ([744, 1439, 56, 56], 'MACHINE_PRINTED_TEXT'), ([370, 1443, 56, 56], 'MACHINE_PRINTED_TEXT'), ([487, 1443, 68, 56], 'MACHINE_PRINTED_TEXT'), ([616, 1447, 56, 48], 'MACHINE_PRINTED_TEXT'), ([3139, 1469, 343, 56], 'MACHINE_PRINTED_TEXT'), ([3827, 1469, 150, 60], 'MACHINE_PRINTED_TEXT'), ([2584, 1477, 27, 38], 'MACHINE_PRINTED_TEXT'), ([2622, 1481, 212, 34], 'HANDWRITING_TEXT'), ([2387, 1477, 60, 63], 'MACHINE_PRINTED_TEXT'), ([2905, 1477, 109, 48], 'MACHINE_PRINTED_TEXT'), ([3173, 1530, 309, 63], 'MACHINE_PRINTED_TEXT'), ([2584, 1538, 374, 45], 'MACHINE_PRINTED_TEXT'), ([2966, 1549, 49, 34], 'MACHINE_PRINTED_TEXT'), ([3861, 1534, 116, 59], 'MACHINE_PRINTED_TEXT'), ([370, 1568, 60, 56], 'MACHINE_PRINTED_TEXT'), ([744, 1568, 56, 56], 'MACHINE_PRINTED_TEXT'), ([495, 1572, 57, 49], 'MACHINE_PRINTED_TEXT'), ([616, 1579, 49, 42], 'MACHINE_PRINTED_TEXT'), ([2387, 1594, 60, 64], 'MACHINE_PRINTED_TEXT'), ([3207, 1594, 143, 60], 'MACHINE_PRINTED_TEXT'), ([3388, 1594, 94, 60], 'MACHINE_PRINTED_TEXT'), ([3895, 1594, 86, 60], 'MACHINE_PRINTED_TEXT'), ([2576, 1602, 64, 52], 'MACHINE_PRINTED_TEXT'), ([2652, 1610, 302, 37], 'MACHINE_PRINTED_TEXT'), ([2966, 1613, 49, 34], 'MACHINE_PRINTED_TEXT'), ([3362, 1621, 22, 18], 'MACHINE_PRINTED_TEXT'), ([2493, 1651, 45, 67], 'MACHINE_PRINTED_TEXT'), ([1413, 1655, 229, 63], 'MACHINE_PRINTED_TEXT'), ([2584, 1663, 370, 49], 'MACHINE_PRINTED_TEXT'), ([2966, 1670, 51, 42], 'MACHINE_PRINTED_TEXT'), ([370, 1696, 158, 56], 'MACHINE_PRINTED_TEXT'), ([646, 1696, 154, 52], 'MACHINE_PRINTED_TEXT'), ([533, 1704, 89, 40], 'MACHINE_PRINTED_TEXT'), ([174, 1711, 59, 64], 'MACHINE_PRINTED_TEXT'), ([2580, 1719, 52, 252], 'MACHINE_PRINTED_TEXT'), ([2659, 1726, 166, 52], 'MACHINE_PRINTED_TEXT'), ([2848, 1726, 166, 52], 'MACHINE_PRINTED_TEXT'), ([2387, 1730, 60, 29], 'MACHINE_PRINTED_TEXT'), ([1413, 1779, 120, 64], 'MACHINE_PRINTED_TEXT'), ([1651, 1779, 112, 64], 'MACHINE_PRINTED_TEXT'), ([926, 1787, 117, 49], 'MACHINE_PRINTED_TEXT'), ([1073, 1799, 34, 26], 'MACHINE_PRINTED_TEXT'), ([370, 1790, 49, 121], 'MACHINE_PRINTED_TEXT'), ([450, 1790, 346, 53], 'MACHINE_PRINTED_TEXT'), ([2659, 1790, 166, 53], 'MACHINE_PRINTED_TEXT'), ([2848, 1790, 162, 56], 'MACHINE_PRINTED_TEXT'), ([2387, 1840, 60, 63], 'MACHINE_PRINTED_TEXT'), ([450, 1855, 346, 52], 'MACHINE_PRINTED_TEXT'), ([2659, 1855, 30, 48], 'MACHINE_PRINTED_TEXT'), ([2708, 1855, 117, 52], 'MACHINE_PRINTED_TEXT'), ([2852, 1855, 14, 44], 'MACHINE_PRINTED_TEXT'), ([1417, 1904, 229, 63], 'MACHINE_PRINTED_TEXT'), ([714, 1919, 44, 56], 'MACHINE_PRINTED_TEXT'), ([450, 1923, 199, 48], 'MACHINE_PRINTED_TEXT'), ([2659, 1923, 166, 44], 'MACHINE_PRINTED_TEXT'), ([370, 1953, 49, 56], 'MACHINE_PRINTED_TEXT'), ([3895, 1972, 86, 63], 'MACHINE_PRINTED_TEXT'), ([1417, 1976, 225, 55], 'MACHINE_PRINTED_TEXT'), ([714, 1979, 48, 56], 'MACHINE_PRINTED_TEXT'), ([2769, 1979, 56, 56], 'MACHINE_PRINTED_TEXT'), ([2961, 1979, 49, 56], 'MACHINE_PRINTED_TEXT'), ([2580, 1983, 56, 52], 'MACHINE_PRINTED_TEXT'), ([450, 1987, 195, 44], 'MACHINE_PRINTED_TEXT'), ([2388, 2010, 56, 159], 'MACHINE_PRINTED_TEXT'), ([2388, 2199, 56, 63], 'MACHINE_PRINTED_TEXT'), ([1413, 2036, 233, 60], 'MACHINE_PRINTED_TEXT'), ([3596, 2036, 230, 63], 'MACHINE_PRINTED_TEXT'), ([2965, 2040, 37, 56], 'MACHINE_PRINTED_TEXT'), ([710, 2044, 52, 55], 'MACHINE_PRINTED_TEXT'), ([2580, 2044, 52, 55], 'MACHINE_PRINTED_TEXT'), ([2773, 2044, 52, 52], 'MACHINE_PRINTED_TEXT'), ([450, 2051, 195, 45], 'MACHINE_PRINTED_TEXT'), ([370, 2055, 49, 78], 'MACHINE_PRINTED_TEXT'), ([2489, 2096, 45, 64], 'MACHINE_PRINTED_TEXT'), ([1413, 2100, 233, 60], 'MACHINE_PRINTED_TEXT'), ([2776, 2104, 45, 56], 'MACHINE_PRINTED_TEXT'), ([2576, 2108, 56, 52], 'MACHINE_PRINTED_TEXT'), ([2965, 2108, 41, 52], 'MACHINE_PRINTED_TEXT'), ([450, 2112, 195, 48], 'MACHINE_PRINTED_TEXT'), ([710, 2112, 45, 48], 'MACHINE_PRINTED_TEXT'), ([283, 2161, 45, 63], 'MACHINE_PRINTED_TEXT'), ([1002, 2169, 136, 49], 'MACHINE_PRINTED_TEXT'), ([1153, 2169, 112, 51], 'MACHINE_PRINTED_TEXT'), ([1651, 2164, 112, 64], 'MACHINE_PRINTED_TEXT'), ([2580, 2168, 52, 56], 'MACHINE_PRINTED_TEXT'), ([374, 2176, 50, 49], 'MACHINE_PRINTED_TEXT'), ([446, 2176, 185, 46], 'MACHINE_PRINTED_TEXT'), ([2773, 2172, 52, 52], 'MACHINE_PRINTED_TEXT'), ([2961, 2172, 45, 56], 'MACHINE_PRINTED_TEXT'), ([672, 2176, 132, 52], 'MACHINE_PRINTED_TEXT'), ([174, 2180, 59, 108], 'MACHINE_PRINTED_TEXT'), ([3596, 2225, 222, 63], 'MACHINE_PRINTED_TEXT'), ([1186, 2229, 82, 56], 'MACHINE_PRINTED_TEXT'), ([1685, 2229, 78, 59], 'MACHINE_PRINTED_TEXT'), ([2580, 2232, 52, 56], 'MACHINE_PRINTED_TEXT'), ([2958, 2232, 52, 60], 'MACHINE_PRINTED_TEXT'), ([374, 2241, 125, 49], 'MACHINE_PRINTED_TEXT'), ([522, 2244, 109, 42], 'MACHINE_PRINTED_TEXT'), ([672, 2240, 136, 52], 'MACHINE_PRINTED_TEXT'), ([1077, 2248, 40, 21], 'MACHINE_PRINTED_TEXT'), ([1428, 2289, 331, 60], 'MACHINE_PRINTED_TEXT'), ([3596, 2293, 222, 56], 'MACHINE_PRINTED_TEXT'), ([2705, 2297, 116, 52], 'MACHINE_PRINTED_TEXT'), ([2829, 2297, 56, 56], 'MACHINE_PRINTED_TEXT'), ([2958, 2297, 52, 56], 'MACHINE_PRINTED_TEXT'), ([182, 2305, 56, 49], 'MACHINE_PRINTED_TEXT'), ([291, 2305, 57, 51], 'MACHINE_PRINTED_TEXT'), ([265, 2320, 15, 15], 'MACHINE_PRINTED_TEXT'), ([359, 2300, 105, 56], 'MACHINE_PRINTED_TEXT'), ([745, 2301, 51, 53], 'MACHINE_PRINTED_TEXT'), ([522, 2305, 98, 49], 'NOISY_TEXT'), ([624, 2305, 60, 49], 'NOISY_TEXT'), ([707, 2312, 23, 19], 'NOISY_TEXT'), ([2584, 2300, 48, 49], 'MACHINE_PRINTED_TEXT'), ([2384, 2312, 63, 158], 'MACHINE_PRINTED_TEXT'), ([2489, 2350, 49, 67], 'MACHINE_PRINTED_TEXT'), ([967, 2353, 298, 60], 'MACHINE_PRINTED_TEXT'), ([1651, 2353, 112, 60], 'MACHINE_PRINTED_TEXT'), ([370, 2368, 271, 45], 'MACHINE_PRINTED_TEXT'), ([684, 2368, 116, 49], 'MACHINE_PRINTED_TEXT'), ([170, 2406, 63, 162], 'MACHINE_PRINTED_TEXT'), ([276, 2410, 52, 71], 'MACHINE_PRINTED_TEXT'), ([1651, 2418, 112, 55], 'MACHINE_PRINTED_TEXT'), ([3706, 2418, 120, 59], 'MACHINE_PRINTED_TEXT'), ([1152, 2421, 113, 56], 'MACHINE_PRINTED_TEXT'), ([1073, 2425, 71, 45], 'MACHINE_PRINTED_TEXT'), ([2784, 2425, 214, 52], 'MACHINE_PRINTED_TEXT'), ([370, 2429, 279, 52], 'MACHINE_PRINTED_TEXT'), ([687, 2433, 113, 52], 'MACHINE_PRINTED_TEXT'), ([1390, 2433, 218, 44], 'MACHINE_PRINTED_TEXT'), ([993, 2482, 151, 56], 'MACHINE_PRINTED_TEXT'), ([1182, 2482, 86, 56], 'MACHINE_PRINTED_TEXT'), ([1681, 2482, 82, 56], 'MACHINE_PRINTED_TEXT'), ([3090, 2490, 1121, 49], 'MACHINE_PRINTED_TEXT'), ([2501, 2494, 555, 41], 'NOISY_TEXT'), ([374, 2493, 290, 52], 'MACHINE_PRINTED_TEXT'), ([680, 2493, 120, 52], 'MACHINE_PRINTED_TEXT'), ([1390, 2493, 215, 48], 'MACHINE_PRINTED_TEXT'), ([922, 2542, 384, 60], 'MACHINE_PRINTED_TEXT'), ([1390, 2542, 377, 60], 'MACHINE_PRINTED_TEXT'), ([2501, 2550, 1083, 56], 'MACHINE_PRINTED_TEXT'), ([382, 2562, 151, 41], 'MACHINE_PRINTED_TEXT'), ([560, 2569, 22, 11], 'MACHINE_PRINTED_TEXT'), ([601, 2569, 15, 30], 'MACHINE_PRINTED_TEXT'), ([745, 2558, 53, 49], 'MACHINE_PRINTED_TEXT'), ([692, 2565, 53, 42], 'MACHINE_PRINTED_TEXT'), ([1190, 2606, 78, 60], 'MACHINE_PRINTED_TEXT'), ([1689, 2606, 74, 56], 'MACHINE_PRINTED_TEXT'), ([2535, 2614, 195, 52], 'MACHINE_PRINTED_TEXT'), ([370, 2618, 271, 48], 'MACHINE_PRINTED_TEXT'), ([687, 2618, 105, 52], 'MACHINE_PRINTED_TEXT'), ([1390, 2618, 218, 48], 'MACHINE_PRINTED_TEXT'), ([2459, 2618, 71, 44], 'MACHINE_PRINTED_TEXT'), ([1186, 2671, 82, 56], 'MACHINE_PRINTED_TEXT'), ([1689, 2671, 70, 56], 'MACHINE_PRINTED_TEXT'), ([2508, 2674, 770, 56], 'MACHINE_PRINTED_TEXT'), ([3936, 2674, 449, 56], 'MACHINE_PRINTED_TEXT'), ([687, 2682, 105, 52], 'MACHINE_PRINTED_TEXT'), ([370, 2686, 275, 48], 'MACHINE_PRINTED_TEXT'), ([1386, 2686, 222, 44], 'MACHINE_PRINTED_TEXT'), ([174, 2724, 61, 87], 'MACHINE_PRINTED_TEXT'), ([174, 2815, 61, 260], 'MACHINE_PRINTED_TEXT'), ([1186, 2735, 82, 56], 'MACHINE_PRINTED_TEXT'), ([1689, 2735, 74, 56], 'MACHINE_PRINTED_TEXT'), ([2504, 2739, 1348, 52], 'MACHINE_PRINTED_TEXT'), ([3929, 2743, 540, 49], 'NOISY_TEXT'), ([438, 2746, 218, 49], 'MACHINE_PRINTED_TEXT'), ([684, 2746, 108, 49], 'MACHINE_PRINTED_TEXT'), ([1386, 2746, 219, 49], 'MACHINE_PRINTED_TEXT'), ([382, 2758, 33, 40], 'MACHINE_PRINTED_TEXT'), ([1651, 2795, 112, 60], 'MACHINE_PRINTED_TEXT'), ([1073, 2799, 192, 52], 'MACHINE_PRINTED_TEXT'), ([3925, 2799, 195, 60], 'MACHINE_PRINTED_TEXT'), ([2501, 2803, 1094, 56], 'MACHINE_PRINTED_TEXT'), ([438, 2810, 147, 53], 'MACHINE_PRINTED_TEXT'), ([593, 2810, 63, 49], 'MACHINE_PRINTED_TEXT'), ([684, 2810, 108, 53], 'MACHINE_PRINTED_TEXT'), ([382, 2814, 37, 45], 'MACHINE_PRINTED_TEXT'), ([280, 2856, 48, 67], 'MACHINE_PRINTED_TEXT'), ([378, 2871, 267, 52], 'MACHINE_PRINTED_TEXT'), ([691, 2875, 109, 52], 'MACHINE_PRINTED_TEXT'), ([691, 2935, 109, 56], 'MACHINE_PRINTED_TEXT'), ([378, 2939, 260, 44], 'MACHINE_PRINTED_TEXT'), ([2387, 2950, 56, 215], 'MACHINE_PRINTED_TEXT'), ([3589, 2954, 902, 67], 'MACHINE_PRINTED_TEXT'), ([382, 2999, 252, 49], 'MACHINE_PRINTED_TEXT'), ([687, 2999, 117, 52], 'MACHINE_PRINTED_TEXT'), ([3864, 3026, 321, 44], 'MACHINE_PRINTED_TEXT'), ([3592, 3029, 256, 41], 'MACHINE_PRINTED_TEXT'), ([4189, 3029, 158, 41], 'MACHINE_PRINTED_TEXT'), ([4355, 3029, 128, 41], 'MACHINE_PRINTED_TEXT'), ([563, 3064, 83, 45], 'MACHINE_PRINTED_TEXT'), ([382, 3068, 79, 41], 'MACHINE_PRINTED_TEXT'), ([480, 3068, 80, 41], 'MACHINE_PRINTED_TEXT'), ([687, 3063, 105, 49], 'MACHINE_PRINTED_TEXT'), ([3592, 3075, 249, 41], 'MACHINE_PRINTED_TEXT'), ([3857, 3075, 278, 37], 'MACHINE_PRINTED_TEXT'), ([4155, 3075, 75, 37], 'MACHINE_PRINTED_TEXT'), ([4238, 3075, 237, 41], 'MACHINE_PRINTED_TEXT'), ([3845, 3116, 362, 60], 'MACHINE_PRINTED_TEXT'), ([4242, 3116, 82, 60], 'MACHINE_PRINTED_TEXT'), ([3585, 3120, 233, 60], 'MACHINE_PRINTED_TEXT'), ([4420, 3124, 67, 52], 'MACHINE_PRINTED_TEXT'), ([378, 3128, 271, 44], 'MACHINE_PRINTED_TEXT'), ([680, 3128, 112, 48], 'MACHINE_PRINTED_TEXT'), ([661, 3147, 11, 18], 'MACHINE_PRINTED_TEXT'), ([371, 3196, 1276, 52], 'MACHINE_PRINTED_TEXT'), ([212, 3204, 121, 38], 'MACHINE_PRINTED_TEXT')]
    
    input_img = cv2.imread(os.path.join(INPUT_PATH, 'test.png'))
    extract_cells = ExtractCells(input_img.copy(), list_xywh_direction)
    list_cells = extract_cells.extract_box_cells()
    
    img_draw_cell = draw_cells(input_img, list_cells)

    cv2.imwrite(os.path.join(OUT_PATH, 'cell.png'), img_draw_cell)
    cv2.imwrite(os.path.join(OUT_PATH, 'test.png'), input_img)


# Remove the text from the small image, keeping the lines
# So this method requires pixellink to do it's part correctly
# def remove_text_image(image, horizontal_size=12, vertical_size=10):
#     image_h, image_w = image.shape
#     horizontal_size = int(image_w * 0.9)
#     vertical_size = int(image_h*0.9)

#     #we dont care about the lines in the middle as it's mostly text
#     # and even when it's a line, it is not important, our algorithm can take care of it
#     image = cv2.rectangle(image, (int(image_w*0.1), int(image_h*0.2)), (int(image_w*0.9), int(image_h*0.8)), (0, 0, 0), cv2.FILLED)
#     vertical = np.copy(image)
#     horizontal = np.copy(image)
    
#     # Create structure element for extracting vertical lines through morphology operations
#     horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
#     horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)

#     # Create structure element for extracting vertical lines through morphology operations
#     verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
#     vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)

#     # return horizontal
#     return horizontal + vertical

# def _check_intersect(box1, cell, flag='vertical'):
#     x_c, y_c, w_c, h_c = cell
#     x_b, y_b, w_b, h_b = box
#     if (x_c < x_b and x_b + w_b < x_c + w_c) and (y_c < y_b and y_b + h_b < y_c + h_c):
#         return True
#     #vertical intersect
#     bb = [box, cell]
#     bb = sorted(bb, key=lambda k: k[0])
#     intersect_w = bb[0][0] + bb[0][2] - bb[1][0]
#     min_width = min(w_c, w_b)
#     #horizontal intersect
#     bb = sorted(bb, key=lambda k: k[1])
#     intersect_h = bb[0][1] + bb[0][3] - bb[1][1]
#     min_height = min(h_c, h_b)
#     if intersect_h / min_height > thresh and intersect_w / min_width > thresh:
#         return True
#     return False