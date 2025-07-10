import torch
import numpy as np
import scipy.ndimage
from PIL import Image, ImageFilter
import time
import hashlib
import io
import pygame  # type: ignore[import]


class AnyType(str): __eq__ = lambda self, __value: True; __ne__ = lambda self, __value: False

def tensor2pil(image: torch.Tensor) -> Image.Image: return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image: Image.Image) -> torch.Tensor: return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
def pil2hex(image: torch.Tensor) -> str: return hashlib.sha256(np.array(tensor2pil(image)).astype(np.uint16).tobytes()).hexdigest()
def pil2mask(image: Image.Image) -> torch.Tensor: inp = np.array(image.convert("L")).astype(np.float32) / 255.0; mask = torch.from_numpy(inp); return 1.0 - mask
def mask_invert(mask:torch.Tensor) -> torch.Tensor: return 1 - mask
def subtract_mask(masks_a:torch.Tensor, masks_b:torch.Tensor) -> torch.Tensor: return torch.clamp(masks_a - masks_b, 0, 255)
def RGB2RGBA(image:Image, mask:Image) -> Image:(R, G, B) = image.convert('RGB').split(); return Image.merge('RGBA', (R, G, B, mask.convert('L')))
def tensor_to_bytes(tensor, quality=80):
    if len(tensor.shape) == 4: tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu().numpy()
    if tensor.dtype != np.uint8: tensor = (tensor * 255).astype(np.uint8)
    pil_image = Image.fromarray(tensor)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='WEBP', quality=quality, method=6)
    return img_byte_arr.getvalue()

def play_sound(sound_type='normal'):
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        if sound_type == 'normal':
            # Frequency = 1000Hz, Duration = 200ms (0.2 second)
            duration = 0.2
            frequency = 1000
            sample_rate = 22050
            frames = int(duration * sample_rate)
            arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
            arr = (arr * 32767).astype(np.int16)
            arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)  # stereo
            sound = pygame.sndarray.make_sound(arr)
            sound.play()
            time.sleep(duration)
        elif sound_type == 'alarm':
            for _ in range(3):
                duration = 1.0
                frequency = 1500
                sample_rate = 22050
                frames = int(duration * sample_rate)
                arr = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))
                arr = (arr * 32767).astype(np.int16)
                arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)  # stereo
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
                time.sleep(duration + 0.2)
        elif sound_type == 'completion':
            melody = [(800,200), (800,200), (800,200), (1000,400), (800,200), (1000,200), (1200,600)]
            for freq, dur in melody:
                duration = dur / 1000.0  # convert ms to seconds
                sample_rate = 22050
                frames = int(duration * sample_rate)
                arr = np.sin(2 * np.pi * freq * np.linspace(0, duration, frames))
                arr = (arr * 32767).astype(np.int16)
                arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)  # stereo
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
                time.sleep(duration + 0.05)
        pygame.mixer.quit()
    except ImportError:
        print(f"Sound alert: {sound_type} (pygame not available)")
    except Exception as e:
        print(f"Sound alert: {sound_type} (error: {e})")

def image2mask(image):
    if image.mode == 'RGBA':
        return torch.tensor([pil2tensor(image.split()[-1])[0, :, :].tolist()])
    elif image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])

def expand_mask(mask:torch.Tensor, grow:int, blur:int) -> torch.Tensor:
    # grow
    c = 0
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in growmask:
        output = m.numpy()
        for _ in range(abs(grow)):
            if grow < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    # blur
    for idx, tensor in enumerate(out):
        pil_image = tensor2pil(tensor.cpu().detach())
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur))
        out[idx] = pil2tensor(pil_image)
    ret_mask = torch.cat(out, dim=0)
    return ret_mask


def resize_preview(image_pil, target_width):
    if image_pil.width <= target_width:return image_pil
    aspect_ratio = image_pil.width / image_pil.height
    new_height = int(target_width / aspect_ratio)
    return image_pil.resize((target_width, new_height), Image.Resampling.LANCZOS)