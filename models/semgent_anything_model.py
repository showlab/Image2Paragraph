from PIL import Image
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class ImageSegmentation():
    def __init__(self, pretrained_model_path, arch="vit_b"):
        self.arch = arch
        self.pretrained_model_path = pretrained_model_path
        self.model = self.initialize_model()

    def initialize_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[self.arch](checkpoint=self.pretrained_model_path)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator
    
    def extract_semantic_from_mask(self, masks):
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        bboxs = []
        object_labels = []
        for mask in sorted_masks:
            m = mask['segmentation']
            bbox = m['bbox']
            object_label = classify_object(m['segmentation') # pass
            bboxs.append(bbox)
            object_labels.append(object_label)
        return bbox, object_labels

    def segment_image(self, image_src):
        image = cv2.imread(image_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.model.generate(image)
        return self.extract_semantic_from_mask(masks)