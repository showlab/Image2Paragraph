from transformers import (CLIPProcessor, CLIPModel, AutoProcessor, CLIPSegForImageSegmentation, 
                          OneFormerProcessor, OneFormerForUniversalSegmentation, 
                          BlipProcessor, BlipForConditionalGeneration)
import torch
import mmcv
import torch.nn.functional as F
import numpy as np
import spacy
from PIL import Image
import pycocotools.mask as maskUtils
from models.segment_models.configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from models.segment_models.configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
# from mmdet.core.visualization.image import imshow_det_bboxes # comment this line if you don't use mmdet

nlp = spacy.load('en_core_web_sm')

class SemanticSegment():
    def __init__(self, device):
        self.device = device
        self.model_init()

    def model_init(self):
        self.init_clip()
        self.init_oneformer_ade20k()
        self.init_oneformer_coco()
        self.init_blip()
        self.init_clipseg()

    def init_clip(self):
        model_name = "openai/clip-vit-large-patch14"
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)

    def init_oneformer_ade20k(self):
        model_name = "shi-labs/oneformer_ade20k_swin_large"
        self.oneformer_ade20k_processor = OneFormerProcessor.from_pretrained(model_name)
        self.oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)

    def init_oneformer_coco(self):
        model_name = "shi-labs/oneformer_coco_swin_large"
        self.oneformer_coco_processor = OneFormerProcessor.from_pretrained(model_name)
        self.oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)

    def init_blip(self):
        model_name = "Salesforce/blip-image-captioning-large"
        self.blip_processor = BlipProcessor.from_pretrained(model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def init_clipseg(self):
        model_name = "CIDAS/clipseg-rd64-refined"
        self.clipseg_processor = AutoProcessor.from_pretrained(model_name)
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device)
        self.clipseg_processor.image_processor.do_resize = False

    @staticmethod
    def get_noun_phrases(text):
        doc = nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def open_vocabulary_classification_blip(self, raw_image):
        captioning_inputs = self.blip_processor(raw_image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**captioning_inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return SemanticSegment.get_noun_phrases(caption)

    def oneformer_segmentation(self, image, processor, model):
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(self.device)
        outputs = model(**inputs)
        predicted_semantic_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        return predicted_semantic_map

    def clip_classification(self, image, class_list, top_k):
        inputs = self.clip_processor(text=class_list, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        if top_k == 1:
            return class_list[probs.argmax().item()]
        else:
            top_k_indices = probs.topk(top_k, dim=1).indices[0]
            return [class_list[index] for index in top_k_indices]

    def clipseg_segmentation(self, image, class_list):
        inputs = self.clipseg_processor(
            text=class_list, images=[image] * len(class_list),
            padding=True, return_tensors="pt").to(self.device)

        h, w = inputs['pixel_values'].shape[-2:]
        fixed_scale = (512, 512)
        inputs['pixel_values'] = F.interpolate(
            inputs['pixel_values'],
            size=fixed_scale,
            mode='bilinear',
            align_corners=False)

        outputs = self.clipseg_model(**inputs)
        logits = F.interpolate(outputs.logits[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        return logits

    
    def semantic_class_w_mask(self, img_src, anns, out_file_name="output/test.json", scale_small=1.2, scale_large=1.6):
        """
        generate class name for each mask
        :param img_src: image path
        :param anns: coco annotations, the same as return dict besides "class_name" and "class_proposals"
        :param out_file_name: output file name
        :param scale_small: scale small
        :param scale_large: scale large
        :return: dict('segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box', "class_name", "class_proposals"})
        """
        img = mmcv.imread(img_src)
        oneformer_coco_seg = self.oneformer_segmentation(Image.fromarray(img), self.oneformer_coco_processor, self.oneformer_coco_model)
        oneformer_ade20k_seg = self.oneformer_segmentation(Image.fromarray(img), self.oneformer_ade20k_processor, self.oneformer_ade20k_model)
        bitmasks, class_names = [], []
        for ann in anns:
        # for ann in anns['annotations']:
            valid_mask = torch.tensor((ann['segmentation'])).bool()
            # valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            coco_propose_classes_ids = oneformer_coco_seg[valid_mask]
            ade20k_propose_classes_ids = oneformer_ade20k_seg[valid_mask]

            top_k_coco_propose_classes_ids = torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
            top_k_ade20k_propose_classes_ids = torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices

            local_class_names = {CONFIG_ADE20K_ID2LABEL['id2label'][str(class_id.item())] for class_id in top_k_ade20k_propose_classes_ids}
            local_class_names.update({CONFIG_COCO_ID2LABEL['refined_id2label'][str(class_id.item())] for class_id in top_k_coco_propose_classes_ids})

            bbox = ann['bbox']
            patch_small = mmcv.imcrop(img, np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]), scale=scale_small)
            patch_large = mmcv.imcrop(img, np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]), scale=scale_large)

            op_class_list = self.open_vocabulary_classification_blip(patch_large)
            local_class_list = list(local_class_names.union(op_class_list))

            top_k = min(len(local_class_list), 3)
            mask_categories = self.clip_classification(patch_small, local_class_list, top_k)
            class_ids_patch_large = self.clipseg_segmentation(patch_large, mask_categories).argmax(0)

            valid_mask_large_crop = mmcv.imcrop(valid_mask.numpy(), np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]), scale=
            scale_large)
            top_1_patch_large = torch.bincount(class_ids_patch_large[torch.tensor(valid_mask_large_crop)].flatten()).topk(1).indices
            top_1_mask_category = mask_categories[top_1_patch_large.item()]

            ann['class_name'] = str(top_1_mask_category)
            ann['class_proposals'] = mask_categories
            class_names.append(ann['class_name'])
            # bitmasks.append(maskUtils.decode(ann['segmentation']))
            bitmasks.append((ann['segmentation']))
        # mmcv.dump(anns, out_file_name)
        return anns
        # below for visualization
        # imshow_det_bboxes(img,
        #             bboxes=None,
        #             labels=np.arange(len(bitmasks)),
        #             segms=np.stack(bitmasks),
        #             class_names=class_names,
        #             font_size=25,
        #             show=False,
        #             out_file='output/result2.png')