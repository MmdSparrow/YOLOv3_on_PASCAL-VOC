import os

# input
DATASET_BASE_PATH = os.path.join("dataset", "archive", "VOCdevkit2007", "VOC2007")
IMAGES_PATH = os.path.join(DATASET_BASE_PATH, "JPEGImages")
IMAGE_SETS_PATH = os.path.join(DATASET_BASE_PATH, "ImageSets", "Main")
ANNOTATIONS_PATH = os.path.join("dataset", "archive", "VOCdevkit2007", "VOC2007", "Annotations")

# classes
CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

