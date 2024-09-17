FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Temel paketlerin kurulumu
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinim dosyasını kopyala ve bağımlılıkları kur
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Tüm proje dosyalarını kopyala
COPY . .

# Ortam değişkenlerini ayarla (Hugging Face token gibi)
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Modeli çalıştır
CMD ["python3", "predict.py"]

