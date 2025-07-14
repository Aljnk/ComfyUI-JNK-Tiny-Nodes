# Standard library imports
import os
import re
import uuid
import json
import shutil
import hashlib
import codecs
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union


import time
import math
import subprocess
from PIL import Image, ImageOps
import json

# Third-party imports
import cv2
import numpy as np
import torch
from google import genai
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from io import BytesIO

# ComfyUI specific imports
import folder_paths  # type: ignore[import]
import server  # type: ignore[import]
from comfy.model_management import InterruptProcessingException  # type: ignore[import]

# Local application imports
from .funs import AnyType, expand_mask, image2mask, pil2tensor, play_sound, resize_preview, subtract_mask, tensor2pil, tensor_to_bytes


any = AnyType("*")



## Models
class LoadModelName_Diff_JNK:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):return {"required": {"model": (folder_paths.get_filename_list("diffusion_models"),),}}
    RETURN_TYPES = (folder_paths.get_filename_list("diffusion_models"), "STRING")
    RETURN_NAMES = ("model", "name",)
    FUNCTION = "load_model_name"
    CATEGORY = "🔧 JNK"
    
    def load_model_name(self, model):
        filename = os.path.splitext(os.path.basename(model))[0]
        return (model, filename)

class LoadModelName_Chpt_JNK:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):return {"required": {"model": (folder_paths.get_filename_list("checkpoints"),),}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "name",)
    FUNCTION = "load_model_name"
    CATEGORY = "🔧 JNK"
    
    def load_model_name(self, model):
       ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", model)
       out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
       filename = os.path.splitext(os.path.basename(model))[0]
       return out[:3] + (filename,)

class LoadLoRAName_JNK:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):return {"required": {"lora_name": (folder_paths.get_filename_list("loras"),),}}
    RETURN_TYPES = (folder_paths.get_filename_list("loras"), "STRING")
    RETURN_NAMES = ("lora_name", "name")
    FUNCTION = "load_lora_name"
    CATEGORY = "🔧 JNK"

    def load_lora_name(self, lora_name):
        filename = os.path.splitext(os.path.basename(lora_name))[0]
        return (lora_name, filename)

## Video
class SaveFrame_JNK:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required":{
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"multiline": True, "default":''}),
                "batch_name": ("STRING", {"default": "Frame"}),
                "frame_number":("INT", {"default": 0, "min": 0, "max": 9999999999}),
			}
		}
    
    CATEGORY = "🔧 JNK"
    RETURN_TYPES = ()
    FUNCTION = "save_img"
    OUTPUT_NODE = True
    type = 'output'

    def save_img(self, image, output_dir, batch_name, frame_number):
        os.makedirs(output_dir, exist_ok=True)
        fname = f'{batch_name}_{str(frame_number).zfill(7)}.png'
        out_fname  = os.path.join(output_dir, fname)
        image = (image[0].clip(0,1)*255.).cpu().numpy().astype('uint8')
        image = Image.fromarray(image)
        image.save(out_fname)
        return ()

class SaveVideoImages_JNK:
    def __init__(self):
        self.type = "output"
        
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required":{
                "images": ("IMAGE",),
                "output_dir": ("STRING", {"multiline": True, "default": 'FOLDER_PATH'}),
                "batch_name": ("STRING",{"default": "IMG"}),
                "frame_tot":("INT",{"default": 8, "min": 0, "max": 9999999999}),
            }
		}
    
    CATEGORY="image"
    RETURN_TYPES = ()
    FUNCTION = "save_vid_img"
    OUTPUT_NODE = True
    
    def save_vid_img(self, images, output_dir, batch_name, frame_tot):
        for frame_number, image in enumerate(images[:frame_tot], start=1):
            fname = f'{batch_name}_{str(frame_number).zfill(7)}.png'
            out_fname  = os.path.join(output_dir, fname)
            if not os.path.exists(out_fname):
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                img.save(out_fname)
        print(f"---JNK---> SaveVideoImages_JNK ---> Saved {frame_tot} images to {output_dir}")
        return ()

## Image
class SaveStaticImage_JNK:
    def __init__(self): self.preview_width = 600
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "file_path": ("STRING", {"multiline": True, "default": ''}),
                "format": (["png", "webp"], {"default": "png"}),
                "quality": ([100, 95, 90, 85, 80], {"default": 80}),
                "skip_if_exists": ("BOOLEAN", {"default": False}),
                "metadata": (["disable", "enable"], {"default": "disable"})
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    CATEGORY = "🔧 JNK"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("success",)
    FUNCTION = "save_stat_img"
    OUTPUT_NODE = True
    type = 'output'
    
    def save_stat_img(self, image, file_path, format="png", quality=80, skip_if_exists=False, metadata="disable", prompt=None, extra_pnginfo=None):
        skip_if_exists = str(skip_if_exists).lower() == "true"
        success = False
        try:
            image_np = (image[0].clip(0,1)*255.).cpu().numpy().astype('uint8')
            image_pil = Image.fromarray(image_np)
            preview_path = os.path.join(folder_paths.get_temp_directory(), f"preview_{uuid.uuid4()}.png")
            preview_image = resize_preview(image_pil, self.preview_width)
            preview_image.save(preview_path)
            
            if skip_if_exists: 
                return {"ui": {"images": [{"filename": os.path.basename(preview_path), "subfolder": "", "type": "temp"}]}, "result": (True,)}

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            exif = image_pil.getexif()
            exif[0x9003] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
            image_pil.info["exif"] = exif.tobytes()
            
            pnginfo = None
            if metadata == "enable":
                pnginfo = PngInfo()
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        pnginfo.add_text(x, json.dumps(extra_pnginfo[x]))

            if format == "png":
                compress_level = int((100 - quality) / 5)
                image_pil.save(file_path, format='PNG', compress_level=compress_level, pnginfo=pnginfo, exif=image_pil.info["exif"])
            else:
                image_pil.save(file_path, format='WEBP', quality=quality, exif=image_pil.info["exif"])
            
            success = True
            return {"ui": {"images": [{"filename": os.path.basename(preview_path), "subfolder": "", "type": "temp"}]}, "result": (success,)}
        
        except Exception:
            print(f"---JNK---> SaveStaticImage_JNK ---> Error saving image: {file_path}")
            return {"ui": {"images": []}, "result": (success,)}

class LoadImageWithCheck_JNK:
    def __init__(self):self.preview_width = 600
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "", "multiline": False}),
                "show_preview": ("BOOLEAN", {"default": True}),
            },
        }
    CATEGORY = "🔧 JNK"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "INT")
    RETURN_NAMES = ("image", "exists", "index")
    FUNCTION = "load_image"
    OUTPUT_NODE = True

    def load_image(self, image_path, show_preview):
        try:
            exists = False;index = 2;results = [];empty_image = torch.zeros((1, 64, 64, 3))
            if image_path and os.path.exists(image_path):
                exists = True;index = 1
                try:
                    with Image.open(image_path) as img:
                        i = np.array(img).astype(np.float32) / 255.0
                        image = torch.from_numpy(i)[None,]
                        if show_preview:
                            img_preview = Image.fromarray(np.clip(i * 255.0, 0, 255).astype(np.uint8))
                            aspect_ratio = img_preview.width / img_preview.height
                            new_height = int(self.preview_width / aspect_ratio)
                            img_preview = img_preview.resize((self.preview_width, new_height), Image.Resampling.LANCZOS)
                            preview_path = os.path.join(folder_paths.get_temp_directory(), f"preview_{uuid.uuid4()}.png")
                            img_preview.save(preview_path)
                            results.append({"filename": os.path.basename(preview_path), "subfolder": "", "type": "temp"})
                except Exception:print(f"---JNK---> LoadImageWithCheck_JNK ---> Error loading image: {image_path}");image = empty_image;exists = False;index = 2
            else:image = empty_image
            if show_preview:
                return {"ui": {"images": results}, "result": (image, exists, index)}
            else:
                return (image, exists, index)
        except Exception:
            print(f"---JNK---> LoadImageWithCheck_JNK ---> Error processing image: {image_path}")
            return (empty_image, False, 2)

class PrepareImageForAI_JNK:
    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "image": ("IMAGE",),
                    "max_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                    "webp_quality": ("INT", {"default": 80, "min": 1, "max": 100, "step": 1}),
                    "show_preview": ("BOOLEAN", {"default": True}),
                },
            }

    CATEGORY = "🔧 JNK"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "quality", "bytes")
    FUNCTION = "doit"

    def doit(self, image, max_size=512, webp_quality=80, show_preview=True):
        print(f"---JNK---> LoadMinmizedImage ---> Started")
        img = tensor2pil(image)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':background.paste(img, mask=img.split()[-1])
            else:background.paste(img, mask=img.split()[-1])
            img = background
        original_size = img.size
        if max(original_size) > max_size:
            ratio = max_size / max(original_size)
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        webp_buffer = BytesIO()
        img.save(webp_buffer, format='WEBP', quality=webp_quality, method=6)
        webp_buffer.seek(0)
        img = Image.open(webp_buffer)
        res = pil2tensor(img.convert('RGB'))
        width = img.size[0];height = img.size[1]
        image_bytes = tensor_to_bytes(res, quality=webp_quality)
        size_bytes = len(image_bytes)
        if size_bytes < 1024:size_formatted = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:size_formatted = f"{size_bytes / 1024:.1f} KB"
        else:size_formatted = f"{size_bytes / (1024 * 1024):.1f} MB"
        results = []
        if show_preview:
            preview_path = os.path.join(folder_paths.get_temp_directory(), f"preview_{uuid.uuid4()}.png")
            preview_image = resize_preview(img, 256)
            preview_image.save(preview_path)
            results.append({"filename": os.path.basename(preview_path), "subfolder": "", "type": "temp"})
        print(f"---JNK---> LoadMinmizedImage ---> Original: {original_size}, Final: {width}x{height}")
        print(f"---JNK---> LoadMinmizedImage ---> Bytes: {size_bytes}, Size: {size_formatted}")
        print(f"---JNK---> LoadMinmizedImage ---> Finished")
        if show_preview:
            return {"ui": {"images": results}, "result": (res, width, height, webp_quality, size_formatted)}
        else:
            return (res, width, height, webp_quality, size_formatted)
    @classmethod
    def IS_CHANGED(self, **kwargs):return float("nan")

class ImageFilterLoader_JNK:
    _cached_images = None
    _cached_paths = None
    _cached_params = None

    def __init__(self) -> None:pass
    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "folder_path": ("STRING", {}),
                "file_filter": ("STRING", {"default": "*.png"}),
                "sort_method": (["numerical", "alphabetical"], {"default": "numerical"}),
                "use_cache": ("BOOLEAN", {"default": False}),
                "skip_img": ("BOOLEAN", {"default": False}),
            },
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE", "STRING", "INT",)
    RETURN_NAMES = ("images", "paths", "total",)
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True, True, False,)
    FUNCTION = "load_image"
    CATEGORY = "🔧 JNK"

    @staticmethod
    def numerical_sort(file_name: Path) -> int:
        subbed = re.sub("\D", "", str(file_name))
        if subbed == "": return 0
        return int(subbed)

    @staticmethod
    def alphabetical_sort(file_name: Path) -> str:
        return str(file_name)

    def load_image(self, folder_path: str, file_filter: str, sort_method: str, use_cache: bool, skip_img: bool) -> Tuple[List[torch.Tensor], List[str]]:
        use_cache = str(use_cache).lower() == "true"
        skip_img = str(skip_img).lower() == "true"
        if use_cache and self._cached_images is not None and self._cached_paths is not None and \
           self._cached_params == (folder_path, file_filter, sort_method, skip_img):
            total_files = len(self._cached_paths)
            if skip_img: return ([], self._cached_paths, total_files)
            return (self._cached_images, self._cached_paths, total_files)
        folder = Path(folder_path)
        if not folder.is_dir():
            raise Exception(f"---JNK---> ImageFilterLoader_JNK ---> Folder path {folder_path} does not exist.")
        sort_method_impl: Callable[[str], Union[int, str]]
        if sort_method == "numerical": sort_method_impl = self.numerical_sort
        elif sort_method == "alphabetical": sort_method_impl = self.alphabetical_sort
        else: raise ValueError(f"Unknown sort method {sort_method}")
        files = sorted(folder.glob(file_filter), key=sort_method_impl)
        if not files:
            raise Exception(f"---JNK---> ImageFilterLoader_JNK ---> No files found in {folder_path} matching filter {file_filter}")
        images = []
        file_paths = []
        for file in files:
            file_paths.append(str(file.absolute()))
            if not skip_img: images.append(pil2tensor(Image.open(file)))
        total_files = len(file_paths)
        if not skip_img: self._cached_images = images
        self._cached_paths = file_paths
        self._cached_params = (folder_path, file_filter, sort_method, skip_img)

        return (images, file_paths, total_files)

class StrokeImage_JNK:
    def __init__(self): self.NODE_NAME = 'JNKStorkeImage'

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "background_image": ("IMAGE", ),
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "stroke_grow": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1}),
                "stroke_width": ("INT", {"default": 8, "min": 0, "max": 999, "step": 1}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "stroke_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "layer_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'stroke_alpha'
    CATEGORY = '🔧 JNK'
    
    def stroke_alpha(self, background_image, layer_image, invert_mask, stroke_grow, stroke_width, blur, stroke_color, layer_mask=None):
        invert_mask = str(invert_mask).lower() == "true"
        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in layer_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
        if layer_mask is not None:
            if layer_mask.dim() == 2:
                layer_mask = torch.unsqueeze(layer_mask, 0)
            l_masks = []
            for m in layer_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        if len(l_masks) == 0:
            print(f"---JNK---> StrokeImage_JNK ---> Error: {self.NODE_NAME} skipped, because the available mask is not found.")
            return (background_image,)

        max_batch = max(len(b_images), len(l_images), len(l_masks))

        grow_offset = int(stroke_width / 2)
        inner_stroke = stroke_grow - grow_offset
        outer_stroke = inner_stroke + stroke_width
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            layer_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            # preprocess
            _canvas = tensor2pil(background_image).convert('RGBA')
            _layer = tensor2pil(layer_image).convert('RGBA')

            if _mask.size != _layer.size:
                _mask = Image.new('L', _layer.size, 'white')
                print(f"---JNK---> StrokeImage_JNK ---> Warning: {self.NODE_NAME} mask size mismatch, using white mask!")

            inner_mask = expand_mask(image2mask(_mask), inner_stroke, blur)
            outer_mask = expand_mask(image2mask(_mask), outer_stroke, blur)
            stroke_mask = subtract_mask(outer_mask, inner_mask)
            color_image = Image.new('RGBA', size=_layer.size, color=stroke_color)
            _canvas.paste(_layer, mask=_mask)
            _canvas.paste(color_image, mask=tensor2pil(stroke_mask))

            ret_images.append(pil2tensor(_canvas))

        print(f"---JNK---> StrokeImage_JNK ---> {self.NODE_NAME} finished processing {max_batch} image(s).")
        return (torch.cat(ret_images, dim=0),)

class AlphaImageNode_JNK:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "custom_width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "custom_height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "fill_type": (["empty", "color"],),
                "color": ("STRING", {"default": "#ffffff"},),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'create_alpha_image'
    CATEGORY = '🔧 JNK'

    def create_alpha_image(self, custom_width, custom_height, fill_type, color):
        color_tuple = (0, 0, 0, 0) if fill_type == "empty" else tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) + (255,)
        ret_image = Image.new('RGBA', (custom_width, custom_height), color_tuple)
        return (pil2tensor(ret_image),)

class AddLayerOverlay_JNK:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_image": ("IMAGE",),
            "overlay_image": ("IMAGE",),
            "position_type": (["top-left", "center"],),
            "center_rotation": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
            "x_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
            "y_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
            "opacity": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 1})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay_image"
    CATEGORY = "🔧 JNK"

    def apply_overlay_image(self, base_image, overlay_image, position_type, center_rotation, x_offset, y_offset, opacity):
        # Prepare all overlay images first
        prepared_overlays = []
        for overlay_tensor in overlay_image:
            overlay_pil = Image.fromarray((255. * overlay_tensor.cpu().numpy()).astype(np.uint8))
            overlay_pil = overlay_pil.convert('RGBA')
            if position_type == "center" and center_rotation != 0:
                overlay_pil = overlay_pil.rotate(center_rotation, expand=True, resample=Image.BICUBIC)
            
            if opacity > 0:
                overlay_array = np.array(overlay_pil)
                overlay_array[..., 3] = overlay_array[..., 3] * (1 - opacity / 100)
                overlay_pil = Image.fromarray(overlay_array)
            
            prepared_overlays.append(overlay_pil)
        
        processed_images = []
        for idx, base_tensor in enumerate(base_image):
            # Determine which overlay to use based on index
            overlay_idx = min(idx, len(prepared_overlays) - 1)  # Use the last overlay if we run out
            overlay_pil = prepared_overlays[overlay_idx]
            
            x = x_offset - overlay_pil.width // 2 if position_type == "center" else x_offset
            y = y_offset - overlay_pil.height // 2 if position_type == "center" else y_offset
            
            base_pil = Image.fromarray((255. * base_tensor.cpu().numpy()).astype(np.uint8))
            base_pil = base_pil.convert('RGBA')
            new_img = Image.new('RGBA', base_pil.size, (0, 0, 0, 0))
            new_img = Image.alpha_composite(new_img, base_pil)
            temp_overlay = Image.new('RGBA', base_pil.size, (0, 0, 0, 0))
            temp_overlay.paste(overlay_pil, (x, y))
            new_img = Image.alpha_composite(new_img, temp_overlay)
            processed_images.append(torch.from_numpy(np.array(new_img).astype(np.float32) / 255.0))

        return (torch.stack(processed_images),)

class GetAlphaLayers_JNK:
    def __init__(self): pass

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("rgb", "mask", "rgba")
    FUNCTION = "extract_regions"
    CATEGORY = "🔧 JNK"

    def find_significant_regions(self, mask: np.ndarray, threshold: float = 0.1, area_ratio_threshold: float = 0.01) -> np.ndarray:
        while len(mask.shape) > 2:
            if mask.shape[0] == 1:mask = mask[0]
            else:return None, 0, None
        significant_mask = (mask > threshold).astype(np.uint8)
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(significant_mask)
        if num_labels <= 1:return None, 0, None
        # Find the largest region
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]  # Skip background
        max_area = max(areas)
        min_area = max_area * area_ratio_threshold
        # Create new labels array with only significant regions
        new_labels = np.zeros_like(labels)
        significant_regions = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                significant_regions.append(i)
                new_labels[labels == i] = len(significant_regions)
        new_num_labels = len(significant_regions) + 1  # +1 for background
        # Create debug visualization
        debug_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i, original_label in enumerate(significant_regions, 1):
            color = np.random.randint(0, 255, 3)
            debug_mask[new_labels == i] = color
        cv2.imwrite('debug_regions.png', debug_mask)
        return new_labels, new_num_labels, stats
        
    def _process_single_image(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # This is the original extract_regions logic
        mask_np = mask.cpu().numpy()
        original_shape = mask_np.shape
        try:
            # Find significant regions with area-based filtering
            labels, num_labels, stats = self.find_significant_regions(mask_np.squeeze())
            if labels is None or num_labels <= 1:
                empty_rgb = torch.ones_like(image)
                empty_mask = torch.zeros_like(mask)
                empty_rgba = torch.zeros((1, *image.shape[:-1], 4))
                return empty_rgb, empty_mask, empty_rgba
            # Create final mask
            final_mask = np.zeros(original_shape, dtype=mask_np.dtype)
            # Find the largest region
            areas = [np.sum(labels == i) for i in range(1, num_labels)]
            largest_region_label = areas.index(max(areas)) + 1
            # Keep only the largest region
            region_mask = (labels == largest_region_label)
            if len(original_shape) == 4:region_mask = region_mask[np.newaxis, np.newaxis, ...]
            elif len(original_shape) == 3:region_mask = region_mask[np.newaxis, ...]
            final_mask[region_mask] = mask_np[region_mask]
            # Convert to tensor
            final_mask_tensor = torch.from_numpy(final_mask).float()
            # Create RGB with white background
            full_rgb = image[..., :3].clone()
            mask_for_rgb = final_mask_tensor.squeeze(0).squeeze(0)
            mask_for_rgb = mask_for_rgb.unsqueeze(0).unsqueeze(-1)
            full_rgb = full_rgb * mask_for_rgb + (1 - mask_for_rgb)
            # Create RGBA
            mask_for_rgba = mask_for_rgb
            full_rgba = torch.cat([image[..., :3], mask_for_rgba], dim=-1)
            return full_rgb, final_mask_tensor, full_rgba
        except Exception as e:
            import traceback
            print(f"---JNK---> GetAlphaLayers_JNK ---> Error: {str(e)}")
            print(f"---JNK---> GetAlphaLayers_JNK ---> Traceback:", traceback.format_exc())
            empty_rgb = torch.ones_like(image)
            empty_mask = torch.zeros_like(mask)
            empty_rgba = torch.zeros((1, *image.shape[:-1], 4))
            return empty_rgb, empty_mask, empty_rgba

    def extract_regions(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Check for batch processing
        batch_size_img = image.shape[0]
        batch_size_mask = mask.shape[0]
        
        # If not in batch mode or single image, use the original processing
        if batch_size_img == 1:
            return self._process_single_image(image, mask)
        
        # Batch processing: use single mask or multiple masks
        use_single_mask = batch_size_mask == 1
        
        all_rgb = []
        all_masks = []
        all_rgba = []
        
        for i in range(batch_size_img):
            # Get current image
            img = image[i:i+1]
            
            # Get appropriate mask
            if use_single_mask:
                msk = mask  # Use the same mask for all images
            else:
                # Use corresponding mask if available, or the last one
                mask_idx = min(i, batch_size_mask - 1)
                msk = mask[mask_idx:mask_idx+1]
            
            # Process single image-mask pair
            try:
                rgb, mask_out, rgba = self._process_single_image(img, msk)
                all_rgb.append(rgb)
                all_masks.append(mask_out)
                all_rgba.append(rgba)
            except Exception as e:
                import traceback
                print(f"---JNK---> GetAlphaLayers_JNK ---> Error: {str(e)}")
                print(f"---JNK---> GetAlphaLayers_JNK ---> Traceback: {traceback.format_exc()}")
                # Use empty tensors on error
                empty_rgb = torch.ones_like(img)
                empty_mask = torch.zeros_like(msk)
                empty_rgba = torch.zeros((1, *img.shape[:-1], 4))
                all_rgb.append(empty_rgb)
                all_masks.append(empty_mask)
                all_rgba.append(empty_rgba)
        
        # Stack results
        combined_rgb = torch.cat(all_rgb, dim=0)
        combined_mask = torch.cat(all_masks, dim=0)
        combined_rgba = torch.cat(all_rgba, dim=0)
        
        return combined_rgb, combined_mask, combined_rgba

class GetAllAlphaLayers_JNK:
    def __init__(self): pass

    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("rgb", "mask", "rgba")
    FUNCTION = "extract_regions"
    CATEGORY = "🔧 JNK"

    def find_significant_regions(self, mask: np.ndarray, threshold: float = 0.1, area_ratio_threshold: float = 0.01) -> np.ndarray:
        while len(mask.shape) > 2:
            if mask.shape[0] == 1:mask = mask[0]
            else:return None, 0, None
        significant_mask = (mask > threshold).astype(np.uint8)
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(significant_mask)
        if num_labels <= 1:return None, 0, None
        # Find the largest region
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]  # Skip background
        max_area = max(areas)
        min_area = max_area * area_ratio_threshold
        # Create new labels array with only significant regions
        new_labels = np.zeros_like(labels)
        significant_regions = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                significant_regions.append(i)
                new_labels[labels == i] = len(significant_regions)
        new_num_labels = len(significant_regions) + 1  # +1 for background
        # Create debug visualization
        debug_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for i, original_label in enumerate(significant_regions, 1):
            color = np.random.randint(0, 255, 3)
            debug_mask[new_labels == i] = color
        cv2.imwrite('debug_regions.png', debug_mask)
        return new_labels, new_num_labels, stats
        
    def _process_single_image(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Modified to keep all regions, not just the largest one
        mask_np = mask.cpu().numpy()
        original_shape = mask_np.shape
        try:
            # Find significant regions with area-based filtering
            labels, num_labels, stats = self.find_significant_regions(mask_np.squeeze())
            if labels is None or num_labels <= 1:
                empty_rgb = torch.ones_like(image)
                empty_mask = torch.zeros_like(mask)
                empty_rgba = torch.zeros((1, *image.shape[:-1], 4))
                return empty_rgb, empty_mask, empty_rgba
            
            # Create final mask - keep all significant regions
            final_mask = np.zeros(original_shape, dtype=mask_np.dtype)
            
            # Keep all regions
            region_mask = (labels > 0)
            if len(original_shape) == 4:
                region_mask = region_mask[np.newaxis, np.newaxis, ...]
            elif len(original_shape) == 3:
                region_mask = region_mask[np.newaxis, ...]
            
            final_mask[region_mask] = mask_np[region_mask]
            
            # Convert to tensor
            final_mask_tensor = torch.from_numpy(final_mask).float()
            
            # Create RGB with white background
            full_rgb = image[..., :3].clone()
            mask_for_rgb = final_mask_tensor.squeeze(0).squeeze(0)
            mask_for_rgb = mask_for_rgb.unsqueeze(0).unsqueeze(-1)
            full_rgb = full_rgb * mask_for_rgb + (1 - mask_for_rgb)
            
            # Create RGBA
            mask_for_rgba = mask_for_rgb
            full_rgba = torch.cat([image[..., :3], mask_for_rgba], dim=-1)
            
            return full_rgb, final_mask_tensor, full_rgba
        except Exception as e:
            print(f"---JNK---> GetAllAlphaLayers_JNK ---> Error: {str(e)}")
            import traceback
            print(f"---JNK---> GetAllAlphaLayers_JNK ---> Traceback: {traceback.format_exc()}")
            empty_rgb = torch.ones_like(image)
            empty_mask = torch.zeros_like(mask)
            empty_rgba = torch.zeros((1, *image.shape[:-1], 4))
            return empty_rgb, empty_mask, empty_rgba

    def extract_regions(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Check for batch processing
        batch_size_img = image.shape[0]
        batch_size_mask = mask.shape[0]
        
        # If not in batch mode or single image, use the original processing
        if batch_size_img == 1:
            return self._process_single_image(image, mask)
        
        # Batch processing: use single mask or multiple masks
        use_single_mask = batch_size_mask == 1
        
        all_rgb = []
        all_masks = []
        all_rgba = []
        
        for i in range(batch_size_img):
            # Get current image
            img = image[i:i+1]
            
            # Get appropriate mask
            if use_single_mask:
                msk = mask  # Use the same mask for all images
            else:
                # Use corresponding mask if available, or the last one
                mask_idx = min(i, batch_size_mask - 1)
                msk = mask[mask_idx:mask_idx+1]
            
            # Process single image-mask pair
            try:
                rgb, mask_out, rgba = self._process_single_image(img, msk)
                all_rgb.append(rgb)
                all_masks.append(mask_out)
                all_rgba.append(rgba)
            except Exception as e:
                print(f"---JNK---> GetAllAlphaLayers_JNK ---> Error processing image {i}: {str(e)}")
                import traceback
                print(f"---JNK---> GetAllAlphaLayers_JNK ---> Traceback:", traceback.format_exc())
                # Use empty tensors on error
                empty_rgb = torch.ones_like(img)
                empty_mask = torch.zeros_like(msk)
                empty_rgba = torch.zeros((1, *img.shape[:-1], 4))
                
                all_rgb.append(empty_rgb)
                all_masks.append(empty_mask)
                all_rgba.append(empty_rgba)
        
        # Stack results
        combined_rgb = torch.cat(all_rgb, dim=0)
        combined_mask = torch.cat(all_masks, dim=0)
        combined_rgba = torch.cat(all_rgba, dim=0)
        
        return combined_rgb, combined_mask, combined_rgba

## Upscale
class TopazPhotoAI_JNK:
    # cd C:\Program Files\Topaz Labs LLC\Topaz Photo AI
    # tpai.exe --help

    '''
    AUTOPILOT SETTINGS
    open TOPAZ PHOTO AI
    Edit > Preferences > AutoPilot Settings
    Change the settings as needed
    '''

    def __init__(self):
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.comfy_dir = os.path.abspath(os.path.join(self.this_dir, '..', '..'))
        self.subfolder = 'upscaled'
        self.output_dir = os.path.join(self.comfy_dir, 'temp')
        self.prefix = 'tpai'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'images': ('IMAGE',),
                'compression': ('INT', {'default': 2,'min': 0,'max': 10,}),
                'tpai_exe': ('STRING', {'default': 'C:\\Program Files\\Topaz Labs LLC\\Topaz Photo AI\\tpai.exe','tooltip': 'Specify the path to the folder that contains the tpai.exe file.'}),
            },
        }

    RETURN_TYPES = ('STRING', 'IMAGE')
    RETURN_NAMES = ('autopilot_settings', 'IMAGE')
    FUNCTION = 'upscale_image'
    CATEGORY = "🔧 JNK"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, True)

    def save_image(self, img, output_dir, filename):
        if not os.path.exists(output_dir):os.makedirs(output_dir)
        file_path = os.path.join(output_dir, filename)
        img.save(file_path)
        return file_path

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image

    def get_settings(self, stdout):
        settings_start = stdout.find('Final Settings for')
        settings_start = stdout.find('{', settings_start)
        count = 0
        settings_end = settings_start
        for i in range(settings_start, len(stdout)):            
            if stdout[i] == '{':count += 1
            elif stdout[i] == '}':count -= 1
            if count == 0:settings_end = i;break
        settings_json = str(stdout[settings_start : settings_end + 1])
        settings = json.loads(settings_json)
        autopilot_settings = settings.pop('autoPilotSettings')
        autopilot_settings_json = json.dumps(autopilot_settings, indent=2).replace('"', "'")
        return autopilot_settings_json

    def topaz_upscale(self, img_file, compression=0, format='png', tpai_exe=None):
        if not os.path.exists(tpai_exe):raise ValueError('Topaz AI Upscaler not found at %s' % tpai_exe)
        if compression < 0 or compression > 10:raise ValueError('compression must be between 0 and 10')        
        target_dir = os.path.join(self.output_dir, self.subfolder)
        if not os.path.exists(target_dir):os.makedirs(target_dir)
        cmd_parts = [f'"{tpai_exe}"','--overwrite',f'--output "{target_dir}"',f'--compression {compression}',f'--format {format}','--showSettings']
        cmd_parts.append(f'"{img_file}"')
        cmd = ' '.join(cmd_parts)
        p_tpai = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        output_file = os.path.join(target_dir, os.path.basename(img_file))
        if not os.path.exists(output_file):shutil.copy(img_file, output_file)
        try:autopilot_settings = self.get_settings(p_tpai.stdout)
        except Exception as e:
            print(f"---JNK---> TopazPhotoAI_JNK ---> Error: {str(e)}")
            autopilot_settings = "{}"
        return (output_file, autopilot_settings)

    def upscale_image(self, images, compression=0, format='png', tpai_exe=None):
        now_millis = int(time.time() * 1000)
        prefix = '%s-%d' % (self.prefix, now_millis)
        upscaled_images = []
        upscale_autopilot_settings = []
        count = 0
        for image in images:
            count += 1
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img_file = self.save_image(img, self.output_dir, '%s-%d.png' % (prefix, count))
            (upscaled_img_file, autopilot_settings) = self.topaz_upscale(img_file, compression, format, tpai_exe=tpai_exe)
            upscaled_image = self.load_image(upscaled_img_file)
            upscaled_images.append(upscaled_image)
            upscale_autopilot_settings.append(autopilot_settings)
        return (upscale_autopilot_settings, upscaled_images)

## Text
class GetTextFromList_JNK:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "text_list_s": ("STRING", {"forceInput": True}),
                "text_list_l": ("LIST", {"forceInput": True}),
            },
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
            }
        }
    
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_text_from_list"
    CATEGORY = "🔧 JNK"

    def flatten_list(self, lst):
        flattened = []
        for item in lst:
            if isinstance(item, list): flattened.extend(self.flatten_list(item))
            else: flattened.append(item)
        return flattened
    
    def get_text_from_list(self, index, text_list_s=None, text_list_l=None):
        text_list = text_list_s if text_list_s is not None else text_list_l
        if text_list is None:return ("",)
        items = self.flatten_list(text_list) if isinstance(text_list, list) else [str(text_list)]
        index = index[0] if isinstance(index, list) and len(index) > 0 else 0
        if index >= len(items):return ("",)
        return (str(items[index]),)

class TextSaver_JNK:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "path": ("STRING", {"default": ""}),
                "overwrite": (["all", "empty", "skip"], {"default": "empty"}),
                "clear": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_text"
    CATEGORY = "🔧 JNK"
    OUTPUT_NODE = True

    def save_text(self, text, path, overwrite, clear):
        if not isinstance(text, str):raise TypeError(f"Expected text to be str, got {type(text).__name__}")
        clear = str(clear).lower() == "true"
        if clear:
            text = re.sub(r'[\"]+', '', text)
            text = re.sub(r'^[.:,\!?\#\-\*\$\@\s]+|[.:,\!?\#\-\*\$\@\s]+$', '', text)
        if os.path.exists(path):
            if overwrite == "skip": return ()
            elif overwrite == "empty":
                if os.path.getsize(path) > 3: return ()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f: f.write(text)
        return ()

class GetSubstring_JNK:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": False}),
                "start": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "length": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "from_end": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_substring"
    CATEGORY = "🔧 JNK"

    def get_substring(self, text: str, start: int, length: int, from_end: bool):
        if from_end: start = len(text) - start - length
        return (text[start:start + length],)

class Model2Key_JNK:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "cut_start": ("INT", {"default": 0, "min": 0, "step": 1}),
                "cut_end": ("INT", {"default": 0, "min": 0, "step": 1}),
                "cut_text": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_text"
    CATEGORY = "🔧 JNK"

    def process_text(self, text, cut_start, cut_end, cut_text):
        for separator in ['\\', '/']:
            if separator in text:text = text.split(separator)[-1]
        name = text
        n_len = len(name)
        if cut_text:
            if cut_text in name:name = name[:name.index(cut_text)]
        if cut_start >= n_len or cut_end >= n_len or  (cut_start+cut_end)>= n_len:
            result=name
        else:
            start_text=name[:cut_start]
            end_text=name[-cut_end:]
            result = f"{start_text}-{end_text}"
        return (result,)

class Text2MD5_JNK:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("md5_hash",)
    FUNCTION = "calculate_md5"
    CATEGORY = "🔧 JNK"

    def calculate_md5(self, text):
        print(f"---JNK---> [JNK MD5] Calculating MD5 for: {text[:50]}... (length: {len(text)})")
        md5_hash = hashlib.md5(text.encode()).hexdigest()
        return (md5_hash,)

class JoinStrings_JNK:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
                "string4": ("STRING", {"default": ""}),
                "string5": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("joined_string",)
    FUNCTION = "join_strings"
    CATEGORY = "🔧 JNK"

    def join_strings(self, string1, string2, string3, string4, string5, delimiter):
        delimiter = delimiter.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        strings = [s for s in [string1, string2, string3, string4, string5] if s]
        result = delimiter.join(strings)
        return (result,)
    

class SplitString_JNK:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "delimiter": ("STRING", {"default": ","}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "skip_every": ("INT", {"default": 0, "min": 0, "max": 10}),
                "max_count": ("INT", {"default": 100, "min": 1, "max": 1000}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_list",)
    FUNCTION = "split_text"
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "🔧 JNK"
    def split_text(self, text, delimiter, start_index, skip_every, max_count):
        if delimiter == "":arr = [text.strip()]
        else:
            delimiter = codecs.decode(delimiter, 'unicode_escape')
            arr = [line.strip() for line in text.split(delimiter) if line.strip()]
        arr = arr[start_index:start_index + max_count * (skip_every + 1):(skip_every + 1)]
        return (arr,)

class GetTimestamp_JNK:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(cls):return {"required": {},}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("timestamp",)
    FUNCTION = "get_timestamp"
    CATEGORY = "🔧 JNK"

    @classmethod
    def IS_CHANGED(self, **kwargs):return float("nan")

    def get_timestamp(self):
        timestamp = str(int(time.time()))
        return (timestamp,)

## Logic
class SwitchInt_JNK:
    def __init__(self):
        self.type = "output"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state":(["ON","OFF"],{"default":"ON"}),
            }
        }

    RETURN_NAMES = ("int",)
    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = "🔧 JNK"

    def convert(self, state):
        if state == "ON":
            idx = 1
        else:
            idx = 0
        return (idx,)

class SwitchIdx_JNK:
    def __init__(self):
        self.type = "output"
		
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state":(["ON","OFF"],{"default":"ON"}),
            }
        }

    RETURN_NAMES = ("int",)
    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = "🔧 JNK"

    def convert(self, state):
        if state == "ON":
            idx = 2
        else:
            idx = 1
        return (idx,)

class GetModels_JNK:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (["unet", "checkpoint"],),
                "search_text": ("STRING", {"default": ""}),
                "model_name": ("STRING", {"default": ""}),
                "mode": (["test", "work"],),
                "seed": ("INT", {"default": 0, "min": 0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 5}),
            }
        }

    RETURN_TYPES = ("INT", folder_paths.get_filename_list("diffusion_models"), folder_paths.get_filename_list("checkpoints"), "STRING", ["unet", "checkpoint"], "STRING", "STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("count", "unet_name", "ckpt_name", "selected_name", "type", "search_text", "found_list", "is_work", "seed")
    FUNCTION = "get_models"
    CATEGORY = "🔧 JNK"

    def get_models(self, type, search_text, model_name, mode, seed, batch_size):
        if type == "unet": model_list = folder_paths.get_filename_list("diffusion_models")
        else: model_list = folder_paths.get_filename_list("checkpoints")
        filtered_models = [model for model in model_list if search_text.lower() in model.lower()]
        count = len(filtered_models)
        is_work=True
        test_seed=0
        if mode == "work":selected_model = next((model for model in filtered_models if model_name.lower() in model.lower()), "")
        else:
            is_work=False
            if filtered_models:
                test_index = (seed // batch_size) % len(filtered_models)
                test_seed=seed % batch_size + 1
                selected_model = filtered_models[test_index]
            else:selected_model = ""
        found_list = "\n".join([f"{i} -> {model}" for i, model in enumerate(filtered_models)])
        if type == "unet":return (count, selected_model, model_list[0] if model_list else "", selected_model, type, search_text, found_list, is_work, test_seed)
        else:return (count, model_list[0] if model_list else "", selected_model, selected_model, type, search_text, found_list, is_work, test_seed)

## System
class BridgeAll_JNK:
    def __init__(self):
        pass
		
    @classmethod
    def INPUT_TYPES(cls):
        return {
             "required": {},
             "optional": {
                "model": ("MODEL",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
            },
        }
        
    RETURN_NAMES = ("model", "pos", "neg", "latent", "vae", "clip", "image",)
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE",)
    FUNCTION = "bridgeall"
    CATEGORY = "🔧 JNK"

    def bridgeall(self, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None):
        return model, pos, neg, latent, vae, clip, image

class QueueStop_JNK():
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "any": (any, ),
                "stop": ("BOOLEAN", {"default": False}),
                "clean_all": ("BOOLEAN", {"default": False}),
                "sound": (["none", "normal", "alarm", "completion"], {"default": "none"}),
                "mode": (["stop", "continue"],),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = 'stop_node'
    CATEGORY = "🔧 JNK"
    OUTPUT_NODE = True

    def stop_node(self, any, stop, clean_all, sound, mode):
        stop = str(stop).lower() == "true"
        clean_all = str(clean_all).lower() == "true"
        if mode == "stop":
            if stop:
                if sound != "none": play_sound(sound)
                if clean_all: server.PromptServer.instance.prompt_queue.wipe_queue()
                raise InterruptProcessingException()
        return (any,)

class CreateFolder_JNK:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "path": ("STRING", {"multiline": True, "default": ''})
            }
        }
    
    CATEGORY = "🔧 JNK"
    RETURN_TYPES = ()
    FUNCTION = "create_folder"
    OUTPUT_NODE = True
    type = 'output'
    
    def create_folder(self, path):
        try:
            directory = os.path.dirname(path) if os.path.basename(path).find('.') != -1 else path
            if directory:os.makedirs(directory, exist_ok=True)
            return {"ui": {"text": f"Directory created: {directory}"}}
        except Exception as e:
            print(f"---JNK---> CreateFolder ---> Error: {str(e)}")
            return {"ui": {"text": f"Error: {str(e)}"}}

class AskGoogleGemini_JNK:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "gemini-2.5-flash"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "rpm": ("INT", {"default": 0, "min": 0, "max": 1000}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "ask_gemini"
    CATEGORY = "🔧 JNK"

    def get_temp_file_path(self, model, api_key):
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        if not os.path.exists(temp_dir):os.makedirs(temp_dir)
        combined_string = f"{model}_{api_key}"
        file_hash = hashlib.md5(combined_string.encode()).hexdigest()[:13]
        return os.path.join(temp_dir, f"jnk_gemini_last_request_{file_hash}.txt")

    def get_last_request_time(self, model, api_key):
        file_path = self.get_temp_file_path(model, api_key)
        try: return float(open(file_path, 'r').read().strip()) if os.path.exists(file_path) else 0
        except:print(f"---JNK---> AskGoogleGemini ---> Error (GLRT) ---> Unable to read temp file: {file_path}");return 0

    def set_last_request_time(self, current_time, model, api_key):
        file_path = self.get_temp_file_path(model, api_key)
        try:
            with open(file_path, 'w') as f:f.write(str(current_time))
        except:print(f"---JNK---> AskGoogleGemini ---> Error (SLRT) ---> Unable to write temp file: {file_path}")

    def handle_rpm_limit(self, rpm, model, api_key):
        if rpm <= 0:return
        time_to_wait = math.ceil(60 / rpm) + 0.1 - time.time() + self.get_last_request_time(model, api_key)
        if time_to_wait > 0: print(f"---JNK---> AskGoogleGemini ---> RPM Pause ({time_to_wait:.2f} sec.)");time.sleep(time_to_wait)
        self.set_last_request_time(time.time(), model, api_key)

    def ask_gemini(self, api_key, model, prompt, rpm, image=None):
        print(f"---JNK---> AskGoogleGemini ---> Started")
        try:
            self.handle_rpm_limit(rpm, model, api_key)
            client = genai.Client(api_key=api_key)
            contents = [prompt]
            if image is not None:
                print(f"---JNK---> AskGoogleGemini ---> Image added")
                image_bytes = tensor_to_bytes(image)
                image_part = genai.types.Part.from_bytes(data=image_bytes, mime_type='image/png')
                contents.append(image_part)
            response = client.models.generate_content(model=model, contents=contents)
            print(f"---JNK---> AskGoogleGemini ---> Finished")
            return (response.text,)
        except Exception as e:print(f"---JNK---> AskGoogleGemini ---> Error (AG): {str(e)}");return (f"Error: {str(e)}",)

class GetGeminiKeys_JNK:
    def __init__(self):pass
    @classmethod
    def INPUT_TYPES(cls):return {"required": {"file_path": ("STRING", {"default": "", "multiline": False}),}}
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("key_1", "key_2", "key_3", "key_4", "key_5")
    FUNCTION = "load_keys"
    CATEGORY = "🔧 JNK"
    def load_keys(self, file_path=""):
        keys = [""] * 5
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for i in range(min(5, len(lines))):
                    key = lines[i].strip()
                    keys[i] = key
        except:print(f"---JNK---> JNKGeminiKeys ---> Error (LK) ---> Unable to read keys from file: {file_path}");pass
        return tuple(keys)
    
class GetGeminiModels_JNK:
    def __init__(self):pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pro": ("STRING", {"default": "gemini-2.5-pro", "multiline": False}),
                "pro_rpm": ("INT", {"default": 5, "min": 0, "max": 1000}),
                "flash": ("STRING", {"default": "gemini-2.5-flash", "multiline": False}),
                "flash_rpm": ("INT", {"default": 10, "min": 0, "max": 1000}),
                "light": ("STRING", {"default": "", "multiline": False}),
                "light_rpm": ("INT", {"default": 15, "min": 0, "max": 1000}),
            }
        }
    RETURN_TYPES = ("STRING", "INT", "STRING", "INT", "STRING", "INT")
    RETURN_NAMES = ("pro", "pro_rpm", "flash", "flash_rpm", "light", "light_rpm")
    FUNCTION = "store_keys"
    CATEGORY = "🔧 JNK"
    def store_keys(self, pro="gemini-2.5-pro", pro_rpm=5, flash="gemini-2.5-flash",flash_rpm=10, light="", light_rpm=15):
        return (pro, pro_rpm, flash, flash_rpm, light, light_rpm)