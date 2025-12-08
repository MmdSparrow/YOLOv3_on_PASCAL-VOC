import os
import xml.etree.ElementTree as ET

from pathlib import Path
from utils.common_utils import convert_rec_cord_to_center_h_w
from configs.common_configs import DATA_PREPROCESS_OUTPUTS_PATH, LABELS_DIR_PATH
from configs.VOC_dataset_configs import IMAGE_SETS_PATH, IMAGES_PATH, CLASS_NAMES, ANNOTATIONS_PATH


class Preprocess:
    def __create_split_file_indexes(self, split_name):
        source_file_path = os.path.join(IMAGE_SETS_PATH, split_name + '.txt')
        output_file_path = os.path.join(DATA_PREPROCESS_OUTPUTS_PATH, split_name + '.txt')
        
        with open(output_file_path, 'w') as output_file:
            with open(source_file_path, 'r') as source_file:
                for basename in source_file:
                    image_name = basename.strip() + '.jpg'
                    full_image_path = os.path.join(IMAGES_PATH, image_name)
                    output_file.write(full_image_path + '\n')
                    
    def __init__(self):
        os.makedirs(LABELS_DIR_PATH, exist_ok=True)
        os.makedirs(DATA_PREPROCESS_OUTPUTS_PATH, exist_ok=True)
        
        self.__create_split_file_indexes('train')
        self.__create_split_file_indexes('val')
        
        with open(os.path.join(DATA_PREPROCESS_OUTPUTS_PATH, 'train.txt'), 'r') as f:
            train_image_paths = f.read().splitlines()
        with open(os.path.join(DATA_PREPROCESS_OUTPUTS_PATH, 'val.txt'), 'r') as f:
            val_image_paths = f.read().splitlines()

        self.all_image_paths = train_image_paths + val_image_paths
        
    def __convert_annotation(self, image_path):
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            xml_path = os.path.join(ANNOTATIONS_PATH, basename + '.xml')
            label_path = os.path.join(LABELS_DIR_PATH, basename + '.txt')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size_elem = root.find('size')
            img_width = int(size_elem.find('width').text)
            img_height = int(size_elem.find('height').text)
            yolo_labels = []
        
            for obj in root.findall('object'):
                difficult = obj.find('difficult')
                if difficult is not None and difficult.text=='0':
                    class_name = obj.find('name').text
                    class_id = CLASS_NAMES.index(class_name)
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    x_center_norm, y_center_norm, width_norm, height_norm= convert_rec_cord_to_center_h_w(xmin, ymin, xmax, ymax, img_width, img_height)
                    yolo_labels.append(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")
                    
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_labels))
                
        except Exception as e:
            print(f"\n[WARNING] Failed to process {image_path}. Error: {e}")
            print("Skipping this file.")
        
    def __create_labels_split_file_indexes(self, split_name):
        with open(os.path.join(DATA_PREPROCESS_OUTPUTS_PATH, split_name + '.txt'), 'r') as fin, open(os.path.join(DATA_PREPROCESS_OUTPUTS_PATH, split_name + '_label.txt'), 'w') as fout:
            for line in fin:
                line = line.strip()
                if line:
                    img_id = Path(line).stem
                    label_path = f"{LABELS_DIR_PATH}/{img_id}.txt"
                    fout.write(label_path + "\n")
                    
    def convert_all_annotations(self):
        for img_path in self.all_image_paths:
            self.__convert_annotation(img_path)
        self.__create_labels_split_file_indexes('train')
        self.__create_labels_split_file_indexes('val')