import torch
import gc
import os
from PIL import Image, ImageEnhance
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from huggingface_hub import login
from controlnet_aux import MLSDdetector, HEDdetector
import numpy as np
from pathlib import Path
import argparse
import random

# Hugging Face API Token'ı ortam değişkeninden al
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN is None:
    raise ValueError("Lütfen HUGGINGFACE_TOKEN ortam değişkenini ayarlayın.")
login(token=HUGGINGFACE_TOKEN)

# PyTorch CUDA bellek yönetimi ayarı
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


# GPU bellek temizleme işlemi (işlemin başında ve sonunda)
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


# Seed ekleyerek aynı sonucu tekrar elde edebilmek için ayar
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Klasörlerin yolları
uploads_dir = "/home/ec2-user/sd-interior-design/uploads"
outputs_dir = "/home/ec2-user/sd-interior-design/outputs"
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# Cihaz kontrolü: GPU varsa 'cuda', yoksa 'cpu' olarak ayarlıyoruz
device = "cuda" if torch.cuda.is_available() else "cpu"

# ControlNet modellerini yükleyin
controlnet_segment = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg",
    torch_dtype=torch.float16,
).to(device)

controlnet_mlsd = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-mlsd",
    torch_dtype=torch.float16,
).to(device)

# Stable Diffusion img2img ControlNet Pipeline'ı yükleyin
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V3.0_VAE",  # Temel model
    controlnet=[controlnet_segment, controlnet_mlsd],
    torch_dtype=torch.float16,
).to(device)

# LoRA ağırlıklarını yükleyelim
lora_weights = "Lam-Hung/controlnet_lora_interior"  # Kullanmak istediğiniz LoRA modeli
pipe.load_lora_weights(lora_weights)

# Xformers'ı etkinleştir
pipe.enable_xformers_memory_efficient_attention()

# Bellek optimizasyonları
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_sequential_cpu_offload()  # Bellek optimizasyonu için ekledik

# ControlNet dedektörlerini doğru repodan yükleyin
mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
hed = HEDdetector.from_pretrained("lllyasviel/Annotators")


# Görüntü işlemleri ve tahminler için fonksiyonlar
class Predictor:
    def resize_to_fixed(self, image, target_size=(1344, 896)):  # Hedef boyut 1344x896
        return image.resize(target_size, Image.Resampling.LANCZOS)

    def enhance_image(self, image):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)  # Görüntünün kontrastını artırıyoruz

    def predict(self, input_image_path, output_image_path, prompt, negative_prompt, num_inference_steps, guidance_scale,
                seed, strength):
        print("Tahmin işlemi başladı...")

        # Belleği temizle
        clear_memory()

        # Seed ayarla
        if seed is not None:
            set_seed(seed)

        img = Image.open(input_image_path)
        img = self.resize_to_fixed(img, target_size=(1344, 896))  # Görüntüyü 1344x896 boyutuna ayarlama

        # ControlNet koşullandırma görüntülerini oluşturma
        mlsd_image = mlsd(img)
        hed_image = hed(img)

        # ControlNet rehberlik ayarları
        control_images = [hed_image, mlsd_image]  # HED ve MLSD koşullandırma
        controlnet_conditioning_scale = [0.5, 0.3]  # HED ve MLSD için

        # Prompt'ları güncelleme
        prompt_with_suffix = prompt  # Yeni prompt kullanıyoruz
        negative_prompt_with_suffix = negative_prompt  # Yeni negatif prompt kullanıyoruz

        # Tahmin işlemi
        with torch.no_grad():  # Belleği daha az kullanmak için
            generated_image = pipe(
                prompt=prompt_with_suffix,
                negative_prompt=negative_prompt_with_suffix,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device).manual_seed(seed) if seed is not None else None,
                image=img,  # Giriş görüntüsü
                control_image=control_images,  # ControlNet koşullandırma görüntüleri
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                strength=strength,  # Strength parametresi
                width=1344,  # Çıktı genişliği
                height=896,  # Çıktı yüksekliği
            ).images[0]

        # Post-processing (Görüntü iyileştirme)
        generated_image = self.enhance_image(generated_image)

        # Çıktının kaydedilmesi
        generated_image.save(output_image_path)

        # İşlem sonrası bellek temizleme
        clear_memory()

        return Path(output_image_path)


# Ana çalıştırma fonksiyonu
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative text prompt")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation")
    parser.add_argument("--strength", type=float, default=0.8, help="Strength for img2img")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    predictor = Predictor()

    # Tahmin işlemini başlat
    output_path = predictor.predict(
        input_image_path=args.input,
        output_image_path=args.output,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        strength=args.strength,
    )

    print(f"Görüntü başarıyla kaydedildi: {output_path}")
