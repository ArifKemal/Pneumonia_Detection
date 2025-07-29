---
title: Zatürre Teşhis Modeli
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: docker
sdk_version: 20.10.23
app_file: app.py
pinned: false
---

# Zatürre Teşhis Modeli

Bu uygulama, göğüs röntgeni görüntülerini analiz ederek zatürre olup olmadığını tahmin eden bir derin öğrenme modelidir.

## 🚀 Özellikler

- **Tek Görüntü Analizi**: Tek bir röntgen görüntüsü analizi
- **Toplu Analiz**: Birden fazla görüntüyü aynı anda analiz etme
- **Analiz Geçmişi**: Önceki analizleri görüntüleme ve takip etme
- **İnteraktif Grafikler**: Zaman içindeki tahmin değerlerini görselleştirme
- **Sonuç İndirme**: Analiz sonuçlarını CSV formatında indirme
- **Güven Eşiği Ayarı**: Kullanıcı tanımlı güven eşiği ile sonuç filtreleme

## 🛠️ Teknik Detaylar

- **Model**: Transfer Learning (VGG16) tabanlı
- **Framework**: TensorFlow/Keras
- **Arayüz**: Streamlit
- **Görselleştirme**: Plotly
- **Doğruluk**: %94.2

## 📊 Model Performansı

- **Doğruluk**: 94.2%
- **Duyarlılık**: 96.8%
- **Özgüllük**: 91.5%
- **F1-Score**: 93.1%

## 🎯 Kullanım

1. **Görüntü Yükle**: Desteklenen formatlar (JPEG, JPG, PNG)
2. **Analiz**: Otomatik görüntü işleme ve tahmin
3. **Sonuç**: Detaylı metrikler ve risk seviyesi
4. **Geçmiş**: Analiz geçmişini takip etme

## ⚠️ Önemli Not

Bu uygulama sadece eğitim ve araştırma amaçlıdır. Tıbbi teşhis için kullanılamaz. Kesin teşhis için mutlaka bir doktora başvurunuz.

## 🔧 Kurulum

### Yerel Kurulum
```bash
# Repository'yi klonla
git clone https://github.com/ArifKemal/Pneumonia_Detection.git
cd Pneumonia_Detection

# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı çalıştır
streamlit run app.py
```

### Streamlit Cloud Deploy
1. Bu repository'yi GitHub'a yükle
2. [Streamlit Cloud](https://share.streamlit.io/)'a git
3. GitHub hesabınla giriş yap
4. "New app" butonuna tıkla
5. Repository'yi seç ve deploy et

## 📁 Proje Yapısı

```
├── app.py                 # Ana uygulama dosyası
├── pneumonia_model.keras  # Eğitilmiş model
├── requirements.txt       # Python bağımlılıkları
└── README.md            # Proje dokümantasyonu
```

## 🎨 Özellikler

- **Modern UI**: Gradient başlıklar ve animasyonlar
- **Responsive Design**: Farklı ekran boyutlarına uyum
- **Dark Theme**: Göz dostu koyu tema
- **Progress Indicators**: İşlem durumu göstergeleri
- **Error Handling**: Kapsamlı hata yönetimi

## 📈 Gelecek Geliştirmeler

- [ ] Daha fazla görüntü formatı desteği
- [ ] Model performans iyileştirmeleri
- [ ] Mobil uygulama entegrasyonu
- [ ] API endpoint'leri
- [ ] Çoklu dil desteği

## 🤝 Katkıda Bulunma

Bu proje eğitim amaçlı geliştirilmiştir. Katkılarınız için teşekkürler!

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.