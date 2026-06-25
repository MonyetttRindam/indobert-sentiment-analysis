# Analisis Emosi Komentar — Kasus Tom Lembong (IndoBERT)

Klasifikasi emosi komentar berbahasa Indonesia ke **5 kelas**: `SADNESS`, `ANGER`,
`SUPPORT`, `HOPE`, `DISAPPOINTMENT`. Model: **IndoBERT** (`indolem/indobertweet-base-uncased`)
yang di-*fine-tune*, dengan metrik utama **Macro F1** (dataset *imbalanced*).

## Struktur

```
app/tomlembong.py        # Aplikasi Streamlit (inferensi, TensorFlow)
streamlit_app.py         # Entry point Streamlit Cloud -> menjalankan app/tomlembong.py
notebook/                # Notebook training (PyTorch) + notebook asli (referensi)
src/                     # Scaffold modul (opsional)
requirements.txt         # Dependency deploy
```

> Catatan framework: **training** dilakukan dengan PyTorch (lihat `notebook/indobert_sentiment.ipynb`),
> sedangkan **app/inferensi** memakai TensorFlow dan memuat model dari Hugging Face Hub
> (`MonyetttRindam/emotion_classification_model`). Bobot lintas-framework, jadi tetap kompatibel.

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub.
2. Di share.streamlit.io, pilih repo + branch.
3. **Main file path:** `streamlit_app.py` (default).
4. **Advanced settings → Python version: `3.11`** (wajib — `tensorflow-cpu==2.15.1`
   hanya punya wheel sampai Python 3.11).
5. Pastikan model HF Hub `MonyetttRindam/emotion_classification_model` **publik**
   (kalau privat, tambahkan `HF_TOKEN` di **Secrets**).

### Catatan deploy
- Free tier RAM ~1 GB. TensorFlow + IndoBERT (~440 MB) cukup berat saat load awal —
  boot pertama lambat dan rawan *out of memory*. Kalau bermasalah, pertimbangkan
  inferensi via PyTorch yang lebih ringan.
- `tensorflow-cpu` dipakai (bukan `tensorflow`) karena Streamlit Cloud tanpa GPU.
- TF `2.15` dipilih agar membawa Keras 2 — menghindari error Keras 3 pada
  `TFAutoModelForSequenceClassification` (kalau pakai TF ≥ 2.16 harus tambah paket `tf-keras`).
