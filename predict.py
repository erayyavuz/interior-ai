# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
import torch
import gc
import numpy as np
from PIL import Image, ImageEnhance
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from huggingface_hub import login
from controlnet_aux import MLSDdetector, HEDdetector
from cog import BasePredictor, Input, Path
from pathlib import Path as SysPath


# Hugging Face API Token'ı ortam değişkeninden al
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    raise ValueError("Hugging Face API token bulunamadı. Lütfen HUGGINGFACE_TOKEN ortam değişkenini ayarlayın.")
else:
    login(token=HUGGINGFACE_TOKEN)


# PyTorch CUDA bellek yönetimi ayarı
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# GPU bellek temizleme işlemi
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


# Seed ekleyerek aynı sonucu tekrar elde edebilmek için ayar
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Predictor(BasePredictor):
    def setup(self):
        """Modeli belleğe yükle"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ControlNet modellerini yükleyin
        self.controlnet_segment = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
        ).to(self.device)

        self.controlnet_mlsd = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float16
        ).to(self.device)

        # Stable Diffusion img2img ControlNet Pipeline'ı yükleyin
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V3.0_VAE",
            controlnet=[self.controlnet_segment, self.controlnet_mlsd],
            torch_dtype=torch.float16,
        ).to(self.device)

        # LoRA ağırlıklarını yükleyelim
        lora_weights = "Lam-Hung/controlnet_lora_interior"
        self.pipe.load_lora_weights(lora_weights)

        # Xformers'ı etkinleştir
        self.pipe.enable_xformers_memory_efficient_attention()

        # Bellek optimizasyonları
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_sequential_cpu_offload()

        # ControlNet dedektörlerini yükleyin
        self.mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
        self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

    @torch.inference_mode()
    def predict(
        self,
        input: Path = Input(description="Input image path"),
        prompt: str = Input(description="Text prompt for image generation"),
        negative_prompt: str = Input(
            description="Negative text prompt", default="low quality, blurry, watermark, unrealistic"
        ),
        num_inference_steps: int = Input(description="Number of inference steps", default=50, ge=1, le=500),
        guidance_scale: float = Input(description="Guidance scale for generation", default=7.5, ge=1, le=50),
        strength: float = Input(description="Strength for img2img", default=0.8, ge=0.0, le=1.0),
        seed: int = Input(description="Random seed for reproducibility", default=None),
    ) -> Path:
        """Giriş verisine göre görüntü oluşturma işlemi başlatılır"""
        clear_memory()

        # Seed ayarla
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        set_seed(seed)

        # Girdi görüntüsünü yükle
        img = Image.open(str(input))
        img = self.resize_to_fixed(img)

        # ControlNet dedektörlerinden koşullandırma görüntülerini oluşturma
        mlsd_image = self.mlsd(img)
        hed_image = self.hed(img)

        # ControlNet koşullandırma ayarları
        control_images = [hed_image, mlsd_image]
        controlnet_conditioning_scale = [0.5, 0.3]

        # Tahmin işlemi
        generated_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            image=img,
            control_image=control_images,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            width=1344,
            height=896,
        ).images[0]

        # Post-processing (Görüntü iyileştirme)
        generated_image = self.enhance_image(generated_image)

        # Çıktının kaydedilmesi
        output_path = SysPath("output.png")
        generated_image.save(output_path)

        clear_memory()

        return Path(output_path)

    def resize_to_fixed(self, image, target_size=(1344, 896)):
        return image.resize(target_size, Image.Resampling.LANCZOS)

    def enhance_image(self, image):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)  # Görüntünün kontrastını artırıyoruz
