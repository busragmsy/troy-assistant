# 🧠 Troy Chatbot  
**E-posta ve PDF tabanlı kurumsal bilgi asistanı (RAG + LLaMA 3.2)**  

![Python Version](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-blue)
![CI/CD](https://github.com/busragmsy/troy-assistant/actions/workflows/ci.yml/badge.svg?branch=main)
![Container](https://img.shields.io/badge/container-gray)
[![GHCR](https://img.shields.io/badge/GHCR-blue)](https://github.com/busragmsy?tab=packages)
![Container](https://img.shields.io/badge/Docker-Ready-blue)
![Database](https://img.shields.io/badge/PostgreSQL-pgvector-lightgrey)


## 🚀 Proje Özeti  

**Troy Chatbot**, kurumsal e-postalar ve PDF dokümanlarından öğrenerek kullanıcılara doğal dilde yanıtlar sunan,  
**RAG (Retrieval-Augmented Generation)** mimarisine sahip bir yapay zekâ destekli asistanıdır.  
Sistem, **Ollama** üzerinde çalışan **LLaMA 3.2 1B** modelini kullanarak hızlı ve hafif bir yanıt motoru sağlar.  

### 🎯 Amaç  
- Dağınık dokümantasyon ve e-posta içeriklerini tek bir bilgi tabanında toplamak  
- Doğal dilde soru-cevap (FAQ, eğitim, hata çözümü vb.)  
- Docker ortamında kolay kurulum ve taşınabilirlik  
- Hibrit arama (vektör + anahtar kelime) ile yüksek doğruluk  

### ⚙️ Kurulum
1️⃣ Gereksinimler
- Docker & Docker Compose
- Python 3.9+
- Ollama (LLaMA 3.2 1B modeli kurulmuş olmalı)
- OpenAI API Key (isteğe bağlı, alternatif LLM için)

### 🧾 Ortam Değişkenleri (.env)  
Proje dizininde `.env` dosyası oluşturun

## 🚀 Çalıştırma

1.  Servisleri Başlatma:
    `docker compose up -d`
2.  Veri Aktarımı:
    `./kb/scripts/pdfs_ingest.sh
./kb/scripts/emails_ingest_ocr.sh`
3.  Projeyi çalıştırmak için:
    `python chat_unified.py --api`
4.  Chatbot Arayüzünü Aç:
    `http://localhost:3000/web/user/index.html`
6.  Admin Panel:
    `http://localhost:3000/web/admin/index.html`
8.  Docker ortamını durdurma:
    `docker compose down`


