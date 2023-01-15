import os
import json
import albumentations
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from taming.data.sflckr import SegmentationBase # for examples included in repo

class Examples(SegmentationBase):
    def __init__(self, size=256, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/coco_examples.txt",
                         data_root="data/coco_images",
                         segmentation_root="data/coco_segmentations",
                         size=size, random_crop=random_crop,
                         interpolation=interpolation,
                         n_labels=183, shift_segmentation=True)
                         

class CocoBase(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
        
def __init__(self, root, annotation, transforms=None):

class CocoImagesAndCaptionsTrain(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, root, annotation, transforms=None):
        super().__init__(size=size,
                         root="data/coco/train2017",
                         annotation="data/coco/annotations/captions_train2017.json")

    def get_split(self):
        return "train"


class CocoImagesAndCaptionsValidation(CocoBase):
    """returns a pair of (image, caption)"""
    """returns a pair of (image, caption)"""
    def __init__(self, root, annotation, transforms=None):
        super().__init__(size=size,
                         root="data/coco/val2017",
                         annotation="data/coco/annotations/captions_val2017.json")
    
    

    def get_split(self):
        return "validation"
