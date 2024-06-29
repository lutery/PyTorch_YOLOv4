import os
import pathlib
import cv2

to_train_parent_path = pathlib.Path(r"F:\Projects\datasets\oc\TK100\data")
to_convert_train_file_path = r"M:\Projects\openSource\python\yolo\pytorch-YOLO-v1\tk100.txt"
converted_train_file_path = r"M:\Projects\openSource\python\yolo\PyTorch_YOLOv4\data\train.txt"
to_convert_test_file_path = r"M:\Projects\openSource\python\yolo\pytorch-YOLO-v1\tk100-test.txt"
converted_test_file_path = r"M:\Projects\openSource\python\yolo\PyTorch_YOLOv4\data\test.txt"

label_file_path = pathlib.Path(r"F:\Projects\datasets\oc\TK100\data\labels")

# # 制作train训练集
# with open(to_convert_train_file_path, 'r') as train_file:
#     with open(converted_train_file_path, 'w') as converted_file:
#         for line in train_file:
#             line = line.strip()
#             if line:
#                 image_info = line.replace('train', 'images').split(' ')
#                 if len(image_info) > 0:
#                     image_path = to_train_parent_path / image_info[0]
#                     if os.path.exists(image_path):
#                         converted_file.write(str(image_path))
#                         converted_file.write('\n')
#                         image = cv2.imread(str(image_path))
#                         image_w, image_h = image.shape[1], image.shape[0]

#                         label_name = image_path.stem + '.txt'
#                         with open(label_file_path / label_name, 'w') as label_file: 
#                             for i in range(0, len(image_info), 5):
#                                 if len(image_info[i+1:i+6]) < 5:
#                                     continue
#                                 x1, y1, x2, y2, classes = image_info[i+1:i+6]
#                                 w, h = int(x2) - int(x1), int(y2) - int(y1)
#                                 centern_x, centern_y = int(x1) + int(w) / 2, int(y1) + int(h) / 2
#                                 label_file.write(f'{classes} {centern_x / image_w} {centern_y / image_h} {w / image_w} {h / image_h}\n')
                            

# 制作测试集合和验证集
with open(to_convert_test_file_path, 'r') as test_file:
    with open(converted_test_file_path, 'w') as converted_file:
        for line in test_file:
            line = line.strip()
            if line:
                image_info = line.replace('test', 'images').split(' ')
                if len(image_info) > 0:
                    image_path = to_train_parent_path / image_info[0]
                    if os.path.exists(image_path):
                        converted_file.write(str(image_path))
                        converted_file.write('\n')
                        image = cv2.imread(str(image_path))
                        image_w, image_h = image.shape[1], image.shape[0]

                        label_name = image_path.stem + '.txt'
                        with open(label_file_path / label_name, 'w') as label_file: 
                            for i in range(0, len(image_info), 5):
                                if len(image_info[i+1:i+6]) < 5:
                                    continue
                                x1, y1, x2, y2, classes = image_info[i+1:i+6]
                                w, h = int(x2) - int(x1), int(y2) - int(y1)
                                centern_x, centern_y = int(x1) + int(w) / 2, int(y1) + int(h) / 2
                                label_file.write(f'{classes} {centern_x / image_w} {centern_y / image_h} {w / image_w} {h / image_h}\n')