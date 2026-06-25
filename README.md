<div align="center">

# 🎭 IndoBERT Emotion Classifier

### Analisis 5 Emosi pada Komentar Media Sosial Berbahasa Indonesia

*Studi kasus: opini publik terhadap kasus Tom Lembong*

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Fine--tuning-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-IndoBERT-FFD21E)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Serving-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)

**[🚀 Live Demo](#)** · **[📓 Notebook](notebook/indobert_sentiment.ipynb)** · **[🤗 Model](https://huggingface.co/MonyetttRindam/emotion_classification_model)**

</div>

---

## 📖 Ringkasan

Sistem **klasifikasi emosi end-to-end** yang mengategorikan komentar Instagram berbahasa Indonesia ke dalam **5 emosi**: `Sadness`, `Anger`, `Support`, `Hope`, dan `Disappointment`.

Proyek ini menutup seluruh siklus *machine learning* — mulai dari **EDA**, **preprocessing** teks media sosial, **fine-tuning IndoBERT**, **evaluasi**, sampai **deployment** sebagai web app interaktif. Dikerjakan untuk kompetisi Kaggle (non-resmi) dengan **Macro F1** sebagai metrik utama karena dataset yang *imbalanced*.

> 💡 **Tantangan utama:** komentar media sosial penuh slang dan emoji, serta distribusi emosi sangat timpang (kelas terkecil hanya ~5% data). Keduanya ditangani lewat preprocessing khusus dan strategi penyeimbangan kelas.

---

## 🏆 Hasil

<div align="center">

| Metric | Score |
|:------:|:-----:|
| **Accuracy** | **88.8%** |
| **Macro F1** | **0.886** |
| **Weighted F1** | 0.886 |

</div>

<details>
<summary><b>📊 Lihat performa per kelas</b></summary>

| Emosi | Precision | Recall | F1-Score |
|:--|:--:|:--:|:--:|
| SADNESS | 0.854 | 0.745 | 0.796 |
| ANGER | 0.826 | 0.844 | 0.835 |
| SUPPORT | 0.923 | 0.929 | 0.926 |
| HOPE | 0.911 | 0.943 | 0.927 |
| DISAPPOINTMENT | 0.920 | 0.979 | 0.949 |

*Dievaluasi pada set validasi (705 sampel).*

</details>

---

## 🖼️ Demo

<div align="center">

| Web App (Streamlit) | Confusion Matrix |
|:--:|:--:|
| <img src="assets/streamlit_demo.png" width="420"/> | <img src="assets/confusion_matrix.png" width="380"/> |
| *Prediksi emosi & confidence real-time* | *Evaluasi 5 kelas pada validasi* |

</div>

> 📌 *Taruh screenshot-mu di folder `assets/` dengan nama `streamlit_demo.png` dan `confusion_matrix.png`.*

---

## ✨ Fitur Utama

- 🔤 **Preprocessing khusus media sosial** — normalisasi slang Indonesia (`gk → tidak`, `yg → yang`) dan pemetaan **emoji → token emosi** (`😢 → [SEDIH]`), karena emoji membawa sinyal emosi yang kuat.
- ⚖️ **Penanganan data imbalanced** — *oversampling* kelas minoritas + **Macro F1** sebagai metrik utama.
- 🧠 **Fine-tuning IndoBERT** — hanya 8 layer encoder terakhir yang di-*unfreeze* agar efisien & stabil pada dataset kecil.
- 🌐 **Web app interaktif** — antarmuka Streamlit yang menampilkan emosi dominan + tingkat keyakinan tiap kelas secara real-time.
- 🔁 **Konsistensi train–serve** — pipeline preprocessing di app identik dengan saat training.

---

## 🛠️ Tech Stack

| Kategori | Tools |
|:--|:--|
| **Modeling** | IndoBERT (`indobertweet-base-uncased`), Hugging Face Transformers, PyTorch |
| **Data & Eval** | pandas, scikit-learn, NumPy |
| **Visualisasi** | Matplotlib, Seaborn |
| **Deployment** | Streamlit, TensorFlow, Hugging Face Hub |

---

## 📂 Struktur Proyek

```
indobert-sentiment-analysis/
├── app/
│   └── tomlembong.py          # Aplikasi Streamlit (inferensi)
├── notebook/
│   └── indobert_sentiment.ipynb   # Pipeline training end-to-end
├── assets/                    # Screenshot untuk README
├── streamlit_app.py           # Entry point Streamlit Cloud
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline

```
Data (komentar IG)
   └─► EDA  ─►  Preprocessing (slang + emoji)  ─►  Oversampling
        └─►  Fine-tune IndoBERT (PyTorch)  ─►  Evaluasi (Macro F1)
              └─►  Deploy: Streamlit + Hugging Face Hub
```

---

## 🚀 Menjalankan Secara Lokal

```bash
# 1. Clone repo
git clone https://github.com/<username>/indobert-sentiment-analysis.git
cd indobert-sentiment-analysis

# 2. Install dependency
pip install -r requirements.txt

# 3. Jalankan app
streamlit run streamlit_app.py
```

<details>
<summary><b>☁️ Catatan deploy ke Streamlit Community Cloud</b></summary>

- **Main file path:** `streamlit_app.py`
- **Python version:** `3.11` (wajib — `tensorflow-cpu==2.15.1` hanya punya wheel s/d Python 3.11)
- Pastikan model di Hugging Face Hub bersifat **publik** (jika privat, tambahkan `HF_TOKEN` pada Secrets)
- Free tier RAM ~1 GB — load awal model cukup berat, mungkin lambat saat cold start

</details>

---

## 📊 Dataset

- **Sumber:** komentar Instagram terkait kasus Tom Lembong (kompetisi Kaggle, non-resmi)
- **Ukuran:** 5.083 data latih · 1.695 data uji
- **Label:** 5 emosi — Sadness, Anger, Support, Hope, Disappointment (*imbalanced*)

---

## 🧩 Detail Model

| Parameter | Nilai |
|:--|:--|
| Base model | `indolem/indobertweet-base-uncased` |
| Trainable layers | 8 encoder layer terakhir + classifier head |
| Learning rate | 3e-5 |
| Batch size | 16 |
| Epochs | 5 (early stopping berbasis Macro F1) |
| Max length | 128 token |

---

<div align="center">

**Dibuat oleh [Nama Kamu](#)** · [LinkedIn](#) · [GitHub](#)

⭐ *Star repo ini jika bermanfaat!*

</div>
