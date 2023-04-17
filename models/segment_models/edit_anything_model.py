import cv2
import torch
import mmcv
import numpy as np
from PIL import Image
from utils.util import resize_long_edge
from concurrent.futures import ThreadPoolExecutor
import time

class EditAnything:
    def __init__(self, image_caption_model):
        self.device = image_caption_model.device
        self.data_type = image_caption_model.data_type
        self.image_caption_model = image_caption_model

    def region_classify_w_blip2(self, images):
        inputs = self.image_caption_model.processor(images=images, return_tensors="pt").to(self.device, self.data_type)
        generated_ids = self.image_caption_model.model.generate(**inputs)
        generated_texts = self.image_caption_model.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [text.strip() for text in generated_texts]

    def process_ann(self, ann, image, target_size=(224, 224)):
        start_time = time.time()
        m = ann['segmentation']
        m_3c = m[:, :, np.newaxis]
        m_3c = np.concatenate((m_3c, m_3c, m_3c), axis=2)
        bbox = ann['bbox']
        region = mmcv.imcrop(image * m_3c, np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]), scale=1)
        resized_region = mmcv.imresize(region, target_size)
        end_time = time.time()
        print("process_ann took {:.2f} seconds".format(end_time - start_time))
        return resized_region, ann

    def region_level_semantic_api(self, image, anns, topk=5):
        """
        rank regions by area, and classify each region with blip2, parallel processing for speed up
        Args:
            image: numpy array
            topk: int
        Returns:
            topk_region_w_class_label: list of dict with key 'class_label'
        """
        start_time = time.time()
        if len(anns) == 0:
            return []
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        topk_anns = sorted_anns[:min(topk, len(sorted_anns))]
        with ThreadPoolExecutor() as executor:
            regions_and_anns = list(executor.map(lambda ann: self.process_ann(ann, image), topk_anns))
        regions = [region for region, _ in regions_and_anns]
        region_class_labels = self.region_classify_w_blip2(regions)
        for (region, ann), class_label in zip(regions_and_anns, region_class_labels):
            ann['class_name'] = class_label
        end_time = time.time()
        print("region_level_semantic_api took {:.2f} seconds".format(end_time - start_time))

        return [ann for _, ann in regions_and_anns]

    def semantic_class_w_mask(self, img_src, anns):
        image = Image.open(img_src)
        image = resize_long_edge(image, 384)
        return self.region_level_semantic_api(image, anns)