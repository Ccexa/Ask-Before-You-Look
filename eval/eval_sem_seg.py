import argparse
import json
import os
import sys
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from tqdm import tqdm

from model.LISA import LISAForCausalLM
from model.COMPASS import CompassForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="COMPASS/LISA batch eval")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--inputs", type=str)
    parser.add_argument("--masks", type=str, help="ground truth mask directory")
    parser.add_argument("--output", type=str)
    parser.add_argument("--output_dir", type=str, default=None, help="directory to save binary masks")
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14")
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--target", type=str, default="tumor")
    parser.add_argument("--VQA", action="store_true", default=False)
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    x = F.pad(x, (0, img_size - w, 0, img_size - h))
    return x


def build_prompt(text, args, conv_type):
    conv = conversation_lib.conv_templates[conv_type].copy()
    conv.messages = []
    p = DEFAULT_IMAGE_TOKEN + "\n" + text
    if args.use_mm_start_end:
        p = p.replace(DEFAULT_IMAGE_TOKEN,
                      DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    conv.append_message(conv.roles[0], p)
    conv.append_message(conv.roles[1], "")
    return conv.get_prompt()


def prepare_image(image_path, args, clip_image_processor, transform):
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image_t = transform.apply_image(image_np)
    resize_list = [image_t.shape[:2]]
    image_t = preprocess(torch.from_numpy(image_t).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda()
    if args.precision == "bf16":
        image_t = image_t.bfloat16()
    elif args.precision == "fp16":
        image_t = image_t.half()
    else:
        image_t = image_t.float()

    return image_clip, image_t, resize_list, original_size_list


def run_model(model, tokenizer, prompt_text, image_clip, image_t, resize_list, original_size_list, max_new_tokens, args):
    prompt = build_prompt(prompt_text, args, args.conv_type)
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
    output_ids, pred_masks = model.evaluate(
        image_clip, image_t, input_ids, resize_list, original_size_list,
        max_new_tokens=max_new_tokens, tokenizer=tokenizer,
    )
    # decode text output
    output_ids_trimmed = output_ids[0][input_ids.shape[1]:]
    text_out = tokenizer.decode(output_ids_trimmed, skip_special_tokens=True).strip()
    return text_out, pred_masks


def mask_has_tumor(mask_path):
    """Return True if the GT mask contains any non-zero (tumor) pixels."""
    if mask_path is None or not os.path.exists(mask_path):
        return False
    mask = np.array(Image.open(mask_path).convert("L"))
    return mask.max() > 0


def save_binary_mask(pred_mask_np, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    binary = (pred_mask_np > 0).astype(np.uint8)
    # print(np.sum(binary))
    img = Image.fromarray(binary)
    palette = [0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254
    img.putpalette(palette)
    img.save(os.path.join(output_dir, filename))


def main(args):
    args = parse_args(args)
    os.makedirs(args.output, exist_ok=True)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.version, cache_dir=None, model_max_length=args.model_max_length,
        padding_side="right", use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.half}.get(args.precision, torch.float32)
    kwargs = {"torch_dtype": torch_dtype}

    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"], load_in_8bit=True,
            ),
        })

    if "lisa" in args.version.lower():
        model = LISAForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower,
            seg_token_idx=args.seg_token_idx, **kwargs)
    else:
        model = CompassForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower,
            seg_token_idx=args.seg_token_idx, **kwargs)
        model.register_hook()

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and not args.load_in_4bit and not args.load_in_8bit:
        import deepspeed
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        model = deepspeed.init_inference(model=model, dtype=torch.half,
                                         replace_with_kernel_inject=True, replace_method="auto").module
        model.model.vision_tower = vision_tower.half().cuda()
    else:
        model = model.float().cuda()

    model.get_model().get_vision_tower().to(device=args.local_rank)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    images = sorted(glob(os.path.join(args.inputs, "*.png")))
    vqa_results = []

    for image_path in tqdm(images):
        filename = os.path.basename(image_path)
        stem = os.path.splitext(filename)[0]
        image_clip, image_t, resize_list, original_size_list = prepare_image(
            image_path, args, clip_image_processor, transform)

        if args.VQA:
            # Step 1: VQA
            vqa_text, _ = run_model(
                model, tokenizer,
                "Is tumor present in the given image?",
                image_clip, image_t, resize_list, original_size_list, max_new_tokens=8, args=args,
            )
            # print(vqa_text)
            # Parse "Yes/No.\n\nDescription"
            parts = vqa_text.split("\n\n", 1)
            yn_part = parts[0].strip().rstrip(".")
            answer = "Yes" if yn_part.lower().startswith("yes") else "No"
            description = parts[1].strip() if len(parts) > 1 else ""
            # print(answer)
            # GT from mask
            gt_mask_path = os.path.join(args.masks, filename) if args.masks else None
            gt = "Yes" if mask_has_tumor(gt_mask_path) else "No"

            item = {
                "image": filename,
                "answer": answer,
                "description": description,
                "gt": gt,
            }

            # Step 2: segmentation only if VQA says Yes
            if answer == "Yes":
                _, pred_masks = run_model(
                    model, tokenizer,
                    "Please segment the tumor.",
                    image_clip, image_t, resize_list, original_size_list, max_new_tokens=512, args=args
                )
                for pred_mask in pred_masks:
                    if pred_mask.shape[0] == 0:
                        continue
                    pred_mask_np = pred_mask.detach().cpu().numpy()[0]
                    if args.output_dir:
                        save_binary_mask(pred_mask_np, args.output_dir, filename)
            elif args.output_dir:
                h, w = original_size_list[0]
                save_binary_mask(np.zeros((h, w), dtype=np.uint8), args.output_dir, filename)

            vqa_results.append(item)

        else:
            # Standard segmentation
            _, pred_masks = run_model(
                model, tokenizer,
                f"Please segment the {args.target}",
                image_clip, image_t, resize_list, original_size_list, max_new_tokens=512, args=args
            )
            for pred_mask in pred_masks:
                if pred_mask.shape[0] == 0:
                    continue
                pred_mask_np = pred_mask.detach().cpu().numpy()[0]
                if args.output_dir:
                    save_binary_mask(pred_mask_np, args.output_dir, filename)

    if args.VQA and vqa_results:
        out_json = os.path.join(args.output, "vqa_results.json")
        with open(out_json, "w") as f:
            json.dump(vqa_results, f, indent=2)
        print(f"VQA results saved to {out_json}")


if __name__ == "__main__":
    main(sys.argv[1:])
