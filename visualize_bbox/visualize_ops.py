import cv2
import os


# COLOR_BOX = (180, 0, 0)
COLOR_BOX = (0, 200, 0)
THICK_BOX = 3
COLOR_TEXT = (0, 0, 255)

IMG_PATH = r'detect.png'
TXT_PATH = r'Tokyo_000_maxmin.txt'
OUT_PATH = r'/home/hhn21/Documents/visualize_bbox/out.png'

def visualize_bbox(img, list_xywh, color_box=COLOR_BOX, color_text=COLOR_TEXT):
    """
    Draw box and number of box to visualize image
    :param color_text:
    :param color_box:
    :param img: raw image to draw
    :param list_xywh: list of xywh format to draw to image
    :return: an drawn image
    """
    img_draw = img.copy()
    for idx, (x, y, w, h) in enumerate(list_xywh):
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color_box, THICK_BOX)

        cv2.putText(img_draw, str(idx), (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_text, 1)
    return img_draw

def get_list_xywh_from_txt(txt_path):
    lines = []
    with open (txt_path, 'r') as f:
        lines = f.read().splitlines()
    list_xywh = []
    for line in lines:
        line = line.split(',')
        x = int(line[0])
        y = int(line[1])
        w = int(line[2]) - int(line[0])
        h = int(line[5]) - int(line[1])
        list_xywh.append([x, y, w, h])
    return list_xywh

# Read the normal img
img = cv2.imread(IMG_PATH)
print(img.shape)

# list_xywh = get_list_xywh_from_txt(TXT_PATH)
list_xywh = [[3746, 3189, 953, 94], [2788, 3189, 955, 94], [2500, 3189, 285, 94], [3746, 3094, 953, 92], [2788, 3094, 955, 92], [2500, 3094, 285, 92], [4464, 2828, 235, 263], [4225, 2828, 236, 263], [3985, 2828, 237, 263], [3746, 2828, 236, 263], [3506, 2828, 237, 263], [3267, 2828, 236, 263], [3027, 2828, 237, 263], [2788, 2828, 236, 263], [2500, 2828, 285, 263], [3746, 2778, 953, 47], [2788, 2778, 955, 47], [2788, 2731, 1911, 44], [2500, 2731, 285, 94], [1740, 3214, 284, 43], [1357, 3213, 284, 44], [1740, 3163, 284, 48], [1644, 3163, 93, 94], [1357, 3163, 284, 47], [1264, 3163, 90, 94], [974, 3163, 287, 94], [878, 3163, 93, 94], [737, 3163, 138, 94], [498, 3163, 236, 94], [255, 3163, 240, 94], [68, 3163, 184, 94], [1740, 3019, 284, 141], [1644, 3019, 93, 141], [1357, 3019, 284, 141], [1264, 3019, 90, 141], [974, 3019, 287, 141], [878, 3019, 93, 141], [737, 3019, 138, 141], [498, 3019, 236, 141], [161, 3019, 334, 141], [68, 3019, 90, 141], [1740, 2877, 284, 139], [1357, 2876, 284, 140], [1740, 2829, 284, 45], [1357, 2829, 284, 44], [974, 2828, 287, 188], [737, 2827, 138, 189], [498, 2827, 236, 189], [207, 2827, 288, 189], [161, 2827, 43, 189], [1740, 2780, 284, 46], [1357, 2780, 284, 46], [1740, 2732, 284, 45], [1357, 2732, 284, 45], [1740, 2684, 284, 45], [1357, 2684, 284, 45], [1740, 2636, 284, 45], [1357, 2635, 284, 46], [1740, 2587, 284, 46], [1357, 2587, 284, 45], [1740, 2539, 284, 45], [1357, 2539, 284, 45], [1740, 2490, 284, 46], [1357, 2490, 284, 46], [1740, 2442, 284, 45], [1357, 2442, 284, 45], [1740, 2394, 284, 45], [1357, 2394, 284, 45], [1740, 2346, 284, 45], [1357, 2346, 284, 45], [1740, 2298, 284, 45], [1357, 2297, 284, 46], [1740, 2249, 284, 46], [1357, 2249, 284, 45], [1740, 2201, 284, 45], [1357, 2201, 284, 45], [1740, 2152, 284, 46], [1644, 2152, 93, 864], [1357, 2152, 284, 46], [1264, 2152, 90, 864], [974, 2152, 287, 673], [878, 2152, 93, 864], [737, 2152, 138, 672], [498, 2152, 236, 672], [161, 2152, 334, 672], [1740, 2105, 284, 44], [1357, 2105, 284, 44], [1740, 2057, 284, 45], [1357, 2056, 284, 46], [1740, 2009, 284, 45], [1357, 2008, 284, 45], [1740, 1961, 284, 45], [1357, 1960, 284, 45], [1740, 1911, 284, 47], [1644, 1911, 93, 238], [1357, 1911, 284, 46], [1264, 1911, 90, 238], [974, 1911, 287, 238], [878, 1911, 93, 238], [737, 1911, 138, 238], [498, 1911, 236, 238], [161, 1911, 334, 238], [68, 1911, 90, 1105], [1740, 1865, 284, 43], [1357, 1864, 284, 44], [1740, 1814, 284, 48], [1644, 1814, 93, 94], [1357, 1814, 284, 47], [1264, 1814, 90, 94], [974, 1814, 287, 94], [878, 1814, 93, 94], [737, 1814, 138, 94], [498, 1814, 236, 94], [161, 1814, 334, 94], [68, 1814, 90, 94], [1740, 1768, 284, 43], [1357, 1768, 284, 43], [1740, 1720, 284, 45], [1357, 1720, 284, 45], [1740, 1672, 284, 45], [1357, 1672, 284, 45], [1740, 1624, 284, 45], [1357, 1624, 284, 45], [1740, 1574, 284, 47], [1644, 1574, 93, 237], [1357, 1574, 284, 47], [1264, 1574, 90, 237], [974, 1574, 287, 237], [878, 1574, 93, 237], [737, 1574, 138, 237], [498, 1574, 236, 237], [161, 1574, 334, 237], [68, 1574, 90, 237], [1740, 1475, 284, 96], [1644, 1475, 93, 96], [1357, 1475, 284, 96], [1264, 1475, 90, 96], [974, 1475, 287, 96], [878, 1475, 93, 96], [737, 1475, 138, 96], [498, 1475, 236, 96], [161, 1475, 334, 96], [68, 1475, 90, 96], [1740, 1432, 284, 40], [1357, 1432, 284, 40], [974, 1431, 287, 41], [737, 1430, 138, 42], [498, 1430, 236, 42], [161, 1430, 334, 42], [1740, 1335, 284, 94], [1644, 1335, 93, 137], [1357, 1335, 284, 94], [1264, 1335, 90, 137], [974, 1335, 287, 93], [878, 1335, 93, 137], [737, 1335, 138, 92], [498, 1335, 236, 92], [161, 1335, 334, 92], [1740, 1288, 284, 44], [1357, 1287, 284, 45], [1740, 1240, 284, 45], [1357, 1239, 284, 45], [1740, 1192, 284, 45], [1357, 1191, 284, 45], [1740, 1144, 284, 45], [1357, 1144, 284, 44], [1357, 1096, 284, 45], [1740, 1095, 284, 46], [1357, 1048, 284, 45], [1740, 1047, 284, 45], [1740, 995, 284, 49], [1644, 995, 93, 337], [1357, 995, 284, 50], [1264, 995, 90, 337], [974, 995, 287, 337], [878, 995, 93, 337], [737, 995, 138, 337], [498, 995, 236, 337], [161, 995, 334, 337], [68, 995, 90, 477], [1740, 952, 284, 40], [1644, 952, 93, 40], [1357, 952, 284, 40], [1264, 952, 90, 40], [974, 952, 287, 40], [878, 952, 93, 40], [1644, 851, 380, 98], [1264, 851, 377, 98], [878, 851, 383, 98], [1264, 804, 760, 44], [878, 804, 383, 44], [737, 804, 138, 188], [498, 804, 236, 188], [68, 804, 427, 188], [1265, 708, 282, 70], [977, 708, 285, 70], [1265, 664, 282, 41], [977, 664, 285, 41], [739, 707, 188, 72], [644, 707, 92, 72], [451, 707, 190, 72], [356, 707, 92, 72], [165, 707, 188, 72], [69, 707, 93, 72], [451, 664, 476, 40], [69, 664, 379, 40], [2031, 709, 138, 70], [1887, 709, 141, 70], [1744, 709, 140, 70], [1600, 709, 141, 70], [1857, 561, 219, 76], [1600, 561, 569, 145], [1857, 400, 219, 158], [1857, 351, 219, 46], [4178, 2348, 286, 335], [4084, 2348, 91, 335], [3796, 2348, 285, 335], [3701, 2348, 92, 335], [3412, 2348, 286, 335], [3318, 2348, 91, 335], [3173, 2348, 142, 335], [2934, 2348, 236, 335], [2600, 2348, 331, 335], [2506, 2348, 91, 335], [4178, 2252, 286, 93], [3796, 2252, 285, 93], [3412, 2251, 286, 94], [2982, 2251, 333, 94], [2600, 2251, 379, 94], [4178, 2155, 286, 94], [4084, 2155, 91, 190], [3796, 2155, 285, 94], [3701, 2155, 92, 190], [3412, 2155, 286, 93], [3318, 2155, 91, 190], [3175, 2155, 140, 93], [2982, 2155, 190, 93], [2600, 2155, 379, 93], [4178, 2107, 286, 45], [3796, 2107, 285, 45], [4178, 2057, 286, 47], [4084, 2057, 91, 95], [3796, 2057, 285, 47], [3701, 2057, 92, 95], [3412, 2057, 286, 95], [3318, 2057, 91, 95], [3175, 2057, 140, 95], [2938, 2057, 234, 95], [2600, 2057, 335, 95], [2506, 2057, 91, 288], [4178, 2010, 286, 44], [3796, 2010, 285, 44], [4178, 1962, 286, 45], [3796, 1962, 285, 45], [4178, 1914, 286, 45], [3796, 1914, 285, 45], [4178, 1866, 286, 45], [3796, 1866, 285, 45], [4178, 1818, 286, 45], [3796, 1818, 285, 45], [4178, 1770, 286, 45], [4084, 1770, 91, 284], [3796, 1770, 285, 45], [3701, 1770, 92, 284], [3412, 1770, 286, 284], [3318, 1770, 91, 284], [3175, 1770, 140, 284], [2938, 1770, 234, 284], [2600, 1770, 335, 284], [2506, 1770, 91, 284], [4178, 1722, 286, 45], [3796, 1722, 285, 45], [4178, 1673, 286, 46], [4084, 1673, 91, 94], [3796, 1673, 285, 46], [3701, 1673, 92, 94], [3412, 1673, 286, 94], [3318, 1673, 91, 94], [3175, 1673, 140, 94], [2938, 1673, 234, 94], [2696, 1673, 239, 94], [2506, 1673, 187, 94], [2938, 1578, 377, 92], [2938, 1530, 235, 45], [2792, 1530, 143, 45], [2649, 1530, 140, 45], [2938, 1481, 235, 46], [2938, 1433, 235, 45], [4181, 1386, 283, 189], [3798, 1386, 283, 189], [3415, 1385, 283, 190], [3176, 1385, 139, 190], [2938, 1385, 235, 45], [2792, 1385, 143, 142], [2649, 1385, 140, 142], [2602, 1385, 333, 285], [4181, 1337, 283, 46], [3798, 1337, 283, 46], [4181, 1289, 283, 45], [3798, 1289, 283, 45], [4181, 1241, 283, 45], [3798, 1241, 283, 45], [4181, 1193, 283, 45], [4084, 1193, 380, 477], [3798, 1193, 283, 45], [3701, 1193, 380, 477], [3415, 1193, 283, 189], [3318, 1193, 380, 477], [3176, 1193, 139, 189], [2938, 1193, 235, 189], [2602, 1193, 333, 189], [4181, 1145, 283, 45], [3798, 1144, 283, 46], [4181, 1097, 283, 45], [3798, 1096, 283, 45], [4181, 1048, 283, 46], [4084, 1048, 94, 142], [3798, 1048, 283, 45], [3701, 1048, 94, 142], [3415, 1048, 283, 142], [3318, 1048, 94, 142], [3176, 1048, 139, 142], [2938, 1048, 235, 142], [2602, 1048, 333, 142], [2506, 1048, 93, 622], [4181, 904, 283, 141], [3798, 904, 283, 141], [3415, 904, 283, 141], [3176, 904, 139, 141], [2938, 904, 235, 141], [2793, 904, 142, 141], [2602, 904, 188, 141], [4181, 855, 283, 46], [3798, 855, 283, 46], [4181, 808, 283, 44], [3798, 807, 283, 45], [4181, 759, 283, 46], [3798, 759, 283, 45], [4181, 711, 283, 45], [3798, 711, 283, 45], [4181, 663, 283, 45], [3798, 663, 283, 45], [4181, 615, 283, 45], [3798, 615, 283, 45], [4181, 567, 283, 45], [3798, 567, 283, 45], [4181, 519, 283, 45], [3798, 519, 283, 45], [4181, 472, 283, 44], [4084, 472, 94, 573], [3798, 472, 283, 44], [3701, 472, 94, 573], [3415, 472, 283, 429], [3318, 472, 94, 573], [3176, 472, 139, 429], [2938, 472, 235, 429], [2602, 472, 333, 429], [2506, 472, 93, 573], [4181, 423, 283, 46], [4084, 423, 94, 46], [3798, 423, 283, 46], [3701, 423, 94, 46], [3415, 423, 283, 46], [3318, 423, 94, 46], [3176, 423, 139, 46], [2938, 423, 235, 46], [2602, 423, 333, 46], [4181, 375, 283, 45], [3798, 375, 283, 45], [4181, 327, 283, 45], [3798, 327, 283, 45], [4181, 279, 283, 45], [3798, 279, 283, 45], [4181, 232, 283, 44], [4084, 232, 94, 188], [3798, 232, 283, 44], [3701, 232, 94, 188], [3415, 232, 283, 188], [3318, 232, 94, 188], [3176, 232, 139, 188], [2938, 232, 235, 188], [2602, 232, 333, 188], [2506, 232, 93, 237], [4181, 185, 283, 44], [4084, 185, 94, 44], [3798, 185, 283, 44], [3701, 185, 94, 44], [3415, 185, 283, 44], [3318, 185, 94, 44], [4084, 89, 380, 93], [3701, 89, 380, 93], [3318, 89, 380, 93], [4084, 40, 380, 46], [3701, 40, 380, 46], [3318, 40, 380, 46], [3176, 40, 139, 189], [2938, 40, 235, 189], [2506, 40, 429, 189], [82, 347, 1321, 293], [76, 26, 1771, 318]]



output_img = visualize_bbox(img, list_xywh)
#write img
cv2.imwrite(OUT_PATH, output_img)