import cv2
import torch
import mmcv
import numpy as np
from PIL import Image
from utils.util import resize_long_edge

class EditAnything:
    def __init__(self, device, image_caption_model):
        self.device = image_caption_model.device
        self.data_type = image_caption_model.data_type
        self.image_caption_model = image_caption_model

    #  working on paraliz these images now
    def region_classify_w_blip2(self, image):
        inputs = self.image_caption_model.processor(images=image, return_tensors="pt").to(self.device, self.data_type)
        generated_ids =  self.image_caption_model.model.generate(**inputs)
        generated_text = self.image_caption_model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def region_level_semantic_api(self, image, anns, topk=5):
        """
        rank regions by area, and classify each region with blip2
        Args:
            image: numpy array
            topk: int
        Returns:
            topk_region_w_class_label: list of dict with key 'class_label'
        """
        topk_region_w_class_label = []
        if len(anns) == 0:
            return []
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        for i in range(min(topk, len(sorted_anns))):
            ann = anns[i]
            m = ann['segmentation']
            m_3c = m[:,:, np.newaxis]
            m_3c = np.concatenate((m_3c,m_3c,m_3c), axis=2)
            bbox = ann['bbox']
            region = mmcv.imcrop(image*m_3c, np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]), scale=1)
            region_class_label = self.region_classify_w_blip2(region)
            ann['class_name'] = region_class_label
            # print(ann['class_label'], str(bbox))
            topk_region_w_class_label.append(ann)
        return topk_region_w_class_label

    def semantic_class_w_mask(self, img_src, anns):
        image = Image.open(img_src)
        image = resize_long_edge(image, 384)
        return self.region_level_semantic_api(image, anns)
