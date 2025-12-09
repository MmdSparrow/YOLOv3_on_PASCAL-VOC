import os
import shutil
import xml.etree.ElementTree as ET

from utils.common_utils import convert_rec_cord_to_center_h_w
from configs.common_configs import IMGES_DIR_PATH, LABELS_DIR_PATH
from configs.VOC_dataset_configs import IMAGE_SETS_PATH, IMAGES_PATH, CLASS_NAMES, ANNOTATIONS_PATH


class PreprocessV2:
    def __init__(self):
        os.makedirs(os.path.join(IMGES_DIR_PATH, 'train'), exist_ok=True)
        os.makedirs(os.path.join(IMGES_DIR_PATH, 'val'), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR_PATH, 'train'), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR_PATH, 'val'), exist_ok=True)
        
    def __convert_annotation(self, basename, split_name):
        try:
            xml_path = os.path.join(ANNOTATIONS_PATH, basename + '.xml')
            label_path = os.path.join(LABELS_DIR_PATH, split_name, basename + '.txt')
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
            print(f"\n[WARNING] Failed to process file with ID {basename}. Error: {e}")
            print("Skipping this file.")
            
    def __split_data(self, split_name, copy_instead_move=False):
        source_file_path = os.path.join(IMAGE_SETS_PATH, split_name + '.txt')
        image_path= os.path.join(IMGES_DIR_PATH, split_name)
        shutil_method= shutil.move
        if copy_instead_move:
            shutil_method= shutil.copy
        with open(source_file_path, 'r') as source_file:
            for basename in source_file:
                basename= basename.strip()
                image_name = basename + '.jpg'
                full_image_path = os.path.join(IMAGES_PATH, image_name)
                shutil_method(full_image_path, image_path)
                self.__convert_annotation(basename, split_name)
    
    def split_all_data(self, copy_instead_move=False):
        self.__split_data('train', copy_instead_move)
        self.__split_data('val', copy_instead_move)