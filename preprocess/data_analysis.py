
import os
# import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


from collections import Counter
from configs.VOC_dataset_configs import ANNOTATIONS_PATH
from configs.common_configs import DATA_ANALYSIS_RESULTS_PATH


class DataAnalysis:
    def __init__(self):
        os.makedirs(DATA_ANALYSIS_RESULTS_PATH, exist_ok=True)
        self.results_file_path= os.path.join(DATA_ANALYSIS_RESULTS_PATH, 'result.txt')
        self.xml_files = [os.path.join(ANNOTATIONS_PATH, f) for f in os.listdir(ANNOTATIONS_PATH) if f.endswith('.xml')]
        
    def class_distribution(self):
        class_counts = Counter()
            
        for xml_file in self.xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                difficult = obj.find('difficult')
                if difficult is None or difficult.text == '0':
                    class_name = obj.find('name').text
                    class_counts[class_name] += 1

        
        file = open(self.results_file_path, 'w')
        try:
            # file.write(f"Total number of files: {len(self.xml_files)}")
            # file.write(f"Total objects found (non-difficult): {sum(class_counts.values())}")
            # file.write("\n Class distribution:")
            # file.write(class_counts)
            
            print(f"Total number of files: {len(self.xml_files)}")
            print(f"Total objects found (non-difficult): {sum(class_counts.values())}")
            print("\n Class distribution:")
            print(class_counts)
        except Exception as e:
            print(f"An error occurred: {e}")        
        finally:
            file.close()
            
        class_df = pd.DataFrame(class_counts.most_common(), columns=["Class", "Count"])

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='Class', data=class_df, palette='viridis')
        plt.title('Class Distribution in PASCAL VOC 2007', fontsize=16)
        plt.xlabel('Number of Objects', fontsize=12)
        plt.ylabel('Class Name', fontsize=12)
        plt.savefig(os.path.join(DATA_ANALYSIS_RESULTS_PATH, 'class_distribution_PASCAL_VOC_2007.png'))


    def images_size(self):
        images_data = []

        for file in self.xml_files:
            tree = ET.parse(file)
            root = tree.getroot()
            file_name = root.find('filename').text
            size_elem = root.find('size')
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
            area = width * height
            images_data.append({
                            'ID': file_name,
                            'width': width,
                            'height': height,
                            'area': area
                        })

        images_df = pd.DataFrame(images_data)
        print('Image sizes DataFrame (5 first rows)')
        print(images_df.head())
        print('\n Image sizes Data Statistics')
        print(images_df.describe())

    def missing_value(self):
        total_files = 0
        negative_image_files = []
        positive_image_files = 0
        for xml_file in self.xml_files:
            total_files += 1
            object_count = 0
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                difficult =obj.find('difficult')
                if difficult is None or difficult.text == '0':
                    object_count += 1
            if object_count == 0: 
                negative_image_files.append(xml_file)
            else:
                positive_image_files += 1

        print(f"Total files checked: {total_files}")
        print(f"Positive files (with objects): {positive_image_files}")
        print(f"Negative files (no objects): {len(negative_image_files)}")
        
        
    def bounding_box_size(self):
        bbox_data = []
        for file in self.xml_files:
            tree = ET.parse(file)
            root = tree.getroot()
            for obj in root.findall('object'):
                difficult = obj.find('difficult')
                if difficult is None or difficult.text == '0':
                    class_name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    bbox_area = bbox_width * bbox_height
                    bbox_data.append({
                            'class': class_name,
                            'width': bbox_width,
                            'height': bbox_height,
                            'area': bbox_area
                        })
        bbox_df = pd.DataFrame(bbox_data)
        print('Bounding box DataFrame (5 first rows)')
        print(bbox_df.head())
        print('\n Bounding box Data Statistics')
        print(bbox_df.describe())