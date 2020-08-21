import numpy as np
import sys
import cv2
import os
import matplotlib.pyplot as plt

INPUT_PATH = r'/home/hhn21/Documents/remove_lines/test'
OUT_PATH = r'/home/hhn21/Documents/remove_lines/result'

def get_mean(lst): 
    if len(lst) ==0:
        return 0
    return sum(lst) / len(lst) 

def get_local_min_index(lst, thresh):
    index = []
    for i, item in enumerate(lst):
        if item < thresh:
            if i == 0 and item <= lst[i+1]:
                index.append(i)
            elif i == len(lst) - 1 and lst[i - 1] >= item:
                index.append(i)
            elif lst[i - 1] > item and item <= lst[i+1]:
                index.append(i)
            elif lst[i - 1] >= item and item < lst[i+1]:
                index.append(i)
    return index

def get_y_roi(horizontal_border_list, img_h):
    y_start = 0
    y_end = img_h
    print('horizontal_border_list:', horizontal_border_list)
    #empty list case
    if len(horizontal_border_list) < 1:
        return y_start, y_end
    
    bottom_border = [item for item in horizontal_border_list if item < img_h/2]
    top_border = [item for item in horizontal_border_list if item > img_h/2]

    if len(bottom_border) == 1:
        y_start = bottom_border[0]
    elif len(bottom_border) > 1:
        for i in range(1, len(bottom_border)-1):
            if bottom_border[i] > bottom_border[i-1] + 1:
                y_start = bottom_border[i-1]
                break
        if y_start == 0:
            y_start = bottom_border[-1]

    if len(top_border) == 1:
        y_end = top_border[0]
    elif len(top_border) > 1:
        for i in range(1, len(top_border)-1):
            if top_border[i] > top_border[i-1] + 1:
                    y_end = top_border[i]
                    break
        if y_end == img_h:
            y_end = top_border[0]

    print ('y_start: ',y_start)
    print ('y_end: ',y_end)
    return y_start, y_end

def remove_vertical_lines(fn, out_dir, binary_img, gray_img):
    input_img = binary_img
    # input_img = binary_img.copy()
    out_img = gray_img.copy()
    img_h, img_w = input_img.shape

    # get the horizontal borders that we care
    horizontal_sum = np.sum(input_img, axis=1)
    horizontal_line_thresh = img_w*0.8*255
    horizontal_border_list = [i for i,item in enumerate(horizontal_sum) if item > horizontal_line_thresh]
    y_start, y_end = get_y_roi(horizontal_border_list, img_h)

    # preprocess the cropped image
    cropped_img = input_img[y_start:y_end,:]
    crop_h, crop_w = cropped_img.shape
    # close_size = int(crop_h*0.18)
    # print('close_size:', close_size)
    # buffer = int(crop_h*0.1)
    # # Morph-close to connect dots and dash and blurred lines
    # close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_size))
    # cropped_img[buffer: crop_h-buffer,:] = cv2.morphologyEx(cropped_img[buffer: crop_h-buffer,:], cv2.MORPH_CLOSE, close_kernel)
    # close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    # cropped_img[0: 3,:] = cv2.morphologyEx(cropped_img[0: 3,:], cv2.MORPH_CLOSE, close_kernel)
    # cropped_img[crop_h-3: crop_h,:] = cv2.morphologyEx(cropped_img[crop_h-3: crop_h,:], cv2.MORPH_CLOSE, close_kernel)


    
    vertical_sum = np.sum(cropped_img, axis=0)
    small_vertical = [item for item in vertical_sum if item < crop_h*0.9*255]
    mean = 1.2*get_mean(small_vertical)
    local_min_index = get_local_min_index(vertical_sum, mean)
    if len(local_min_index) < 2:
        return out_img
    
    line_thresh = crop_h*0.9*255
    y_pos = np.arange(crop_w)
    bar_chart = plt.bar(y_pos,vertical_sum)
    roi_start = 0
    roi_end = 0
    slices = []
    for i in range(len(local_min_index)-1):
        line_count = 0
        x_start = local_min_index[i] +1
        x_end = local_min_index[i+1]
        for j in range(x_start,x_end):
            if vertical_sum[j] > line_thresh:
                line_count += 1
        thickness = x_end - x_start
        if (thickness < 4 and line_count > 0) or (thickness < 5 and line_count > 1) or (thickness < 15 and line_count > thickness*0.7):
            for bar in range(x_start,x_end):
                bar_chart[bar].set_color('r')
            roi_end = x_start-3
            if roi_end < 0:
                continue
            slices.append(out_img[:, roi_start:roi_end])
            roi_start = x_end-1
    slices.append(out_img[:, roi_start:img_w])


    out_img = np.concatenate(slices, 1)
    plt.xticks(y_pos, '')
    plt.axis('off')
    plt.title('')
    plt.axhline(y=mean,linewidth=1, color='k')
    plt.axhline(y=line_thresh,linewidth=1, color='g')
    fn = fn.replace('.png', '_plt.png')
    plt.savefig(os.path.join(out_dir, fn))
    plt.close()
    return out_img

def no_lines_img(fn, out_dir, img):
    pad_color = [0, 0, 0]
    """
    :param img:
    :return: the img without it's lines
    """
    # Make img gray
    if len(img.shape) != 2:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    # Blur and make binary
    blur_img = cv2.bilateralFilter(gray_img, 9, 15, 15)
    not_img = cv2.bitwise_not(blur_img)
    binary_img = cv2.adaptiveThreshold(not_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, close_kernel)
    binary_img = cv2.copyMakeBorder(binary_img, 0, 0, 2, 2, cv2.BORDER_CONSTANT, value=pad_color)
    no_lines_img = remove_vertical_lines(fn, out_dir, binary_img, gray_img)
    return no_lines_img, binary_img


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
            print(fn)

    return files_img

def main():
    input_folders = [f.path for f in os.scandir(INPUT_PATH) if f.is_dir()]
    for input_folder in input_folders:
        for fn in sorted(get_list_files(input_folder)):
            # if fn != 'dash6.png':
            # if fn != 'dash3.png':
                # continue
            input_folder_name = os.path.basename(input_folder)
            out_dir = os.path.join(OUT_PATH, input_folder_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            input_img = cv2.imread(os.path.join(input_folder, fn))
            print('===============================================')
            print(input_folder)
            print(fn)
            print(input_img.shape)
            out_img, binary_img = no_lines_img(fn, out_dir, input_img)
            print('out_img.shape')
            print(out_img.shape)

            fn = fn.replace('.jpg', '.png')
            fn = fn.replace('.jpeg', '.png')
            cv2.imwrite(os.path.join(out_dir, fn.replace('.png',"_out.png")), out_img)
            cv2.imwrite(os.path.join(out_dir, fn.replace('.png',"_binary.png")), binary_img)
            cv2.imwrite(os.path.join(out_dir, fn), input_img)

if __name__ == "__main__":
    main()