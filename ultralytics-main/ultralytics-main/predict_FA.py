from ultralytics import YOLO
from ensemble_boxes import *
import os

# 创建 YOLO 模型
model = YOLO("/ensemble_models/yolov8x_rong.pt")
model.predict(source="/detect/input", conf=0.10, save=True, save_conf=True, save_txt=True, augment=True, name='ouput_B_x_010')

model = YOLO("/ensemble_models/yolov8s_rong.pt")
model.predict(source="/detect/input", conf=0.10, save=True, save_conf=True, save_txt=True, augment=True, name='ouput_B_s_010')

model = YOLO("/ensemble_models/yolov8n_yuan.pt")
model.predict(source="/detect/input", conf=0.10, save=True, save_conf=True, save_txt=True, augment=True, name='ouput_B_n_010')

# Function to search for a file in a folder
def find_file_in_folder(filename, folder):
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        return file_path
    return ""


folder1 = "runs/detect/ouput_B_x_010/labels"
folder2 = "runs/detect/ouput_B_s_010/labels"
folder3 = "runs/detect/ouput_B_n_010/labels"

# Specify your three folders
folder_paths = [folder1, folder2, folder3]

# Initialize an empty list to store the data matrices
data_matrices = []


# 将YOLO格式转换为WBF格式（适用于三层嵌套的情况）
def convert_yolo_to_wbf(yolo_boxes_list):
    wbf_boxes_list = []
    for file_boxes in yolo_boxes_list:
        wbf_boxes = []
        for box in file_boxes:
            x_center, y_center, width, height = map(float, box)
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            wbf_boxes.append([x_min, y_min, x_max, y_max])
        wbf_boxes_list.append(wbf_boxes)
    return wbf_boxes_list


# 将WBF格式的boxes转换为YOLO格式
def convert_wbf_boxes_to_yolo(wbf_boxes):
    yolo_boxes = []
    for box in wbf_boxes:
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        yolo_boxes.append([x_center, y_center, width, height])
    return yolo_boxes


# Read output.txt and process each line
with open("output.txt", "r") as file:
    for line in file:
        # Remove newline character and any leading/trailing whitespaces
        filename = line.strip()

        cnt = 0
        # Search for the file in each folder
        data_lists = []  # Initialize an empty list for data in each folder
        for folder_path in folder_paths:
            result_path = find_file_in_folder(filename, folder_path)

            # Check if the file is found in this folder
            if result_path:
                cnt += 1
                # Read and process your data from the file
                # Split each line into a list using space as a delimiter
                with open(result_path, "r") as txt_file:
                    data_lists = [line.strip().split() for line in txt_file.readlines()]
            data_matrices.append(data_lists)
        # print(data_matrices)
        if cnt == 0:
            continue
        boxes_list_yolo = []
        for file_lists in data_matrices:
            file_boxes_yolo = []
            for data_list in file_lists:
                extracted_numbers = data_list[1:5]
                file_boxes_yolo.append(extracted_numbers)
            boxes_list_yolo.append(file_boxes_yolo)
        print(boxes_list_yolo)

        scores_list = []
        for file_lists in data_matrices:
            file_boxes_yolo = []
            for data_list in file_lists:
                extracted_numbers = data_list[5]
                file_boxes_yolo.append(extracted_numbers)
            scores_list.append(file_boxes_yolo)
        # print(scores_list)

        labels_list = []
        for file_lists in data_matrices:
            file_boxes_yolo = []
            for data_list in file_lists:
                extracted_numbers = data_list[0]
                file_boxes_yolo.append(extracted_numbers)
            labels_list.append(file_boxes_yolo)
        # print(labels_list)

        # 将boxes_list_yolo转换为weighted_boxes_fusion所需的格式
        boxes_list_wbf = convert_yolo_to_wbf(boxes_list_yolo)
        # 现在boxes_list_wbf具有weighted_boxes_fusion函数所需格式
        print(boxes_list_wbf)

        weights = [1] * cnt
        iou_thr = 0.5
        skip_box_thr = 0.0001
        sigma = 0.1
        scores_list = [[float(score) for score in scores] for scores in scores_list]
        boxes, scores, labels = weighted_boxes_fusion(boxes_list_wbf, scores_list, labels_list, weights=weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        print(boxes, scores, labels)

        # 将WBF输出的boxes转换为YOLO格式
        yolo_boxes_from_wbf = convert_wbf_boxes_to_yolo(boxes)

        # 打印转换后的YOLO格式的boxes
        print(yolo_boxes_from_wbf)

        result_matrices = []
        for file_lists in labels:
            result = [int(file_lists)]
            result_matrices.append(result)
        print(result_matrices)

        i = 0
        for file_lists in yolo_boxes_from_wbf:
            result = file_lists
            for j in result:
                result_matrices[i].append(j)
            i += 1
        print(result_matrices)

        output_filepath = f"runs/detect/B_labels/{filename}"

        with open(output_filepath, "w") as output_file:
            for row in result_matrices:
                # 将每个元素转换为字符串并用空格连接
                line = " ".join(map(str, row))
                # 将行写入文件
                output_file.write(line + "\n")

        data_matrices.clear()

# Now data_matrices contains the data for each file found in each folder
# You can further process the data_matrices according to your needs