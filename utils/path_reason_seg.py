import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
# from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)


class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        # self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)


        # reason_seg_data, splits = reason_seg_data.split("|")
        # splits = splits.split("_")
        # images = []
        # for split in splits:
        #     images_split = glob.glob(
        #         os.path.join(
        #             base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
        #         )
        #     )
        #     images.extend(images_split)
        # jsons = [path.replace(".jpg", ".json") for path in images]
        # self.reason_seg_data = (images, jsons)

        # print("number of reason_seg samples: ", len(images))

        dataset_name, split = reason_seg_data.split("|")
        self.data_json = json.load(open(os.path.join(base_image_dir, dataset_name, split, "data.json"), 'r'))
        self.images = [os.path.join(base_image_dir, dataset_name, split, 'images', item['filename']) for item in self.data_json]
        self.masks = [os.path.join(base_image_dir, dataset_name, split, 'annotations', item['filename']) for item in self.data_json]

        assert len(self.images) == len(self.masks), "number of images and masks should be the same"
        assert len(self.data_json) == len(self.images), "number of data_json and images should be the same"

        print("number of reason_seg samples: ", len(self.images))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # images, jsons = self.reason_seg_data
        idx = random.randint(0, len(self.images) - 1)
        image_path = self.images[idx]
        mask_path = self.masks[idx]


        label = Image.open(mask_path)
        label = np.array(label)
        label[label==0] = 255
        label -= 1
        label[label==254] = 255
        unique_label = np.unique(label).tolist()
        if 255 in unique_label:
            unique_label.remove(255)
        if len(unique_label) == 0:
            return self.__getitem__(0)

        
        # img = cv2.imread(image_path)
        # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
    
        item = self.data_json[idx]

        questions = [DEFAULT_IMAGE_TOKEN + "\n" + item['query']]
        answers = [random.choice(ANSWER_LIST) + " " + item['explanation']]
        conversations = []

        conv = conversation_lib.default_conversation.copy()
       
        assert len(questions) == len(answers), "number of questions and answers should be the same"
        assert len(questions) == 1, "number of questions should be 1"
        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        sampled_classes = conversations
        
        masks = []
        for class_id in unique_label:
            masks.append(label == class_id)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )

