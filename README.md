# Dokumentasi Teknis: Sistem Rekomendasi Olahraga (Hybrid CF + CBF)

Dokumen ini menjelaskan secara rinci aspek teknis dari implementasi Sistem Rekomendasi Olahraga yang menggunakan pendekatan hybrid Collaborative Filtering (CF) dan Content-Based Filtering (CBF) dengan model Deep Learning.

**Daftar Isi:**
1.  [Pendahuluan](#1-pendahuluan)
2.  [Arsitektur Sistem](#2-arsitektur-sistem)
3.  [Sumber Data](#3-sumber-data)
4.  [Pra-pemrosesan Data](#4-pra-pemrosesan-data)
5.  [Pemodelan](#5-pemodelan)
    * [5.1. Model Hybrid (Collaborative Filtering + Content-Based) - Deep Learning](#51-model-hybrid-collaborative-filtering--content-based---deep-learning)
    * [5.2. Matriks Similaritas Content-Based (TF-IDF & Cosine Similarity)](#52-matriks-similaritas-content-based-tf-idf--cosine-similarity)
6.  [Pelatihan Model Hybrid](#6-pelatihan-model-hybrid)
7.  [Proses Rekomendasi Hybrid](#7-proses-rekomendasi-hybrid)
8.  [Penyimpanan Model ke TensorFlow.js](#8-penyimpanan-model-ke-tensorflowjs)
9.  [Struktur Kode dan Fungsi Utama](#9-struktur-kode-dan-fungsi-utama)
10. [Ketergantungan (Dependencies)](#10-ketergantungan-dependencies)
11. [Panduan Setup (Singkat)](#11-panduan-setup-singkat)

---

## 1. Pendahuluan

Proyek ini bertujuan untuk membangun sistem rekomendasi olahraga yang dapat memberikan saran jenis latihan kepada pengguna berdasarkan histori aktivitas mereka, profil pengguna, dan kemiripan antar jenis olahraga. Sistem ini mengimplementasikan metode hybrid yang menggabungkan kekuatan Collaborative Filtering (memanfaatkan interaksi user-item) dan Content-Based Filtering (memanfaatkan deskripsi item) melalui model Jaringan Syaraf Tiruan (Deep Learning).

---

## 2. Arsitektur Sistem

Alur kerja sistem secara umum adalah sebagai berikut:
1.  **Pemuatan Data**: Mengambil dataset pengguna, histori olahraga, dan detail olahraga dari sumber.
2.  **Analisis Data Eksploratif (EDA)**: Memahami struktur dan karakteristik data.
3.  **Pra-pemrosesan Data**:
    * Penggabungan dataset.
    * Perhitungan fitur baru (misal: kalori terbakar).
    * Encoding fitur kategorikal (ID pengguna, ID olahraga, nama latihan, fitur pengguna lainnya).
    * Normalisasi fitur numerik.
    * Pembentukan data input (X) dan target (y) untuk model.
    * Pembagian data menjadi set pelatihan dan validasi.
4.  **Pembuatan Matriks Content-Based**:
    * Menggunakan TF-IDF pada deskripsi olahraga (gabungan nama latihan dan kalori/jam).
    * Menghitung cosine similarity antar olahraga untuk mendapatkan matriks similaritas item-item.
5.  **Pembangunan dan Pelatihan Model Hybrid Deep Learning**:
    * Mendefinisikan arsitektur model Keras yang menerima input ID pengguna, ID olahraga, ID jenis latihan (sebagai fitur konten), dan fitur tambahan (durasi ternormalisasi, fitur pengguna yang sudah di-OHE).
    * Melatih model untuk memprediksi skor interaksi (dalam hal ini, `kalori_terbakar_normal`).
6.  **Generasi Rekomendasi Hybrid**:
    * Untuk seorang pengguna, prediksi skor dilakukan menggunakan model Deep Learning (komponen CF).
    * Skor dari matriks Content-Based (kemiripan dengan olahraga yang pernah dilakukan pengguna) juga dihitung.
    * Kedua skor ini digabungkan dengan bobot tertentu (`alpha`) untuk menghasilkan skor hybrid final.
    * Memberikan top-K rekomendasi berdasarkan skor hybrid.
7.  **Ekspor Model**: Menyimpan model Keras yang telah dilatih ke format TensorFlow.js untuk potensi deployment di web.

---

## 3. Sumber Data

Sistem menggunakan tiga dataset utama yang dimuat dari repositori GitHub:
* Base URL: `https://raw.githubusercontent.com/hilmanfaujiabdilah/capstone-rekomendasi-olahraga/refs/heads/main/dataset%20olahraga`

    1.  **`olahraga.csv`**: Berisi informasi detail mengenai jenis-jenis olahraga.
        * Kolom penting: `id_olahraga`, `latihan` (nama latihan), `kalori_jam`, `Tingkat`.
    2.  **`history_olahraga.csv`**: Berisi catatan histori aktivitas olahraga pengguna.
        * Kolom penting: `id_history`, `id_user`, `id_olahraga`, `tanggal_olahraga`, `durasi_menit`.
    3.  **`user_dataset.csv`**: Berisi informasi profil pengguna.
        * Kolom penting: `id_user`, `umur`, `jenis_kelamin`, `tinggi_badan`, `berat_badan`, `target_berat`, `aktivitas_harian`, `rutin_olahraga`, `ngemil_malam`, `jam_tidur`, `motivasi`.

---

## 4. Pra-pemrosesan Data

Langkah-langkah pra-pemrosesan data dilakukan oleh serangkaian fungsi:

* **`merge_data(df_history, df_user, df_olahraga)`**:
    * Menggabungkan `df_history` dengan `df_user` (mengambil `aktivitas_harian`, `rutin_olahraga`) berdasarkan `id_user`.
    * Hasilnya digabungkan lagi dengan `df_olahraga` (mengambil `kalori_jam`) berdasarkan `id_olahraga`.

* **`compute_calories(df)`**:
    * Menambahkan kolom `kalori_terbakar` yang dihitung dengan formula:
        `(durasi_menit / 60) * kalori_jam`.

* **`encode_ids(df)`**:
    * Melakukan Label Encoding untuk `id_user` dan `id_olahraga`.
    * Menghasilkan kolom baru `user_encoded` dan `olahraga_encoded`.
    * Mengembalikan DataFrame yang telah diupdate dan dictionary mapping untuk encoding dan decoding ID.

* **`encode_latihan(df, df_olahraga)`**:
    * Membuat Label Encoding untuk nama latihan (`latihan`).
    * Menghasilkan kolom baru `latihan_encoded`.
    * Mengembalikan DataFrame yang telah diupdate dan dictionary mapping.

* **`one_hot_encode_features(df, fitur_kategori)`**:
    * Melakukan One-Hot Encoding pada kolom-kolom yang ada di list `fitur_kategori` (contoh: `['aktivitas_harian', 'rutin_olahraga']`).
    * Menggunakan `sklearn.preprocessing.OneHotEncoder`.
    * Mengembalikan DataFrame dengan kolom OHE baru, objek encoder, dan list nama kolom hasil OHE.

* **`normalize_features(df, kolom_durasi, kolom_kalori)`**:
    * Melakukan Min-Max Normalisasi pada kolom durasi (default: `durasi_menit`) dan kalori (default: `kalori_terbakar`).
    * Menghasilkan kolom `durasi_normal` dan `kalori_terbakar_normal`.
    * Mengembalikan DataFrame yang telah diupdate serta nilai min/max untuk durasi dan kalori (untuk keperluan denormalisasi atau inferensi).

* **`build_X_y(df, nama_kol_input, nama_kol_target)`**:
    * Membentuk matriks fitur `X_all` dari kolom-kolom yang didefinisikan dalam `nama_kol_input`.
    * Membentuk vektor target `y_all` dari kolom `nama_kol_target` (default: `kalori_terbakar_normal`).
    * Input untuk model (`nama_kol_input`) terdiri dari: `['user_encoded', 'olahraga_encoded', 'latihan_encoded', 'durasi_normal']` + kolom hasil OHE.

* **`split_train_val(X_all, y_all, frac_train)`**:
    * Membagi `X_all` dan `y_all` menjadi set pelatihan dan validasi dengan proporsi `frac_train` (default: 0.8 untuk data latih).

* **Pengumpulan Metadata**:
    * Semua dictionary mapping hasil encoding, objek OneHotEncoder, dan nilai min/max normalisasi dikumpulkan dalam satu dictionary `metadata`. Ini penting untuk proses inferensi di masa depan.

---

## 5. Pemodelan

Sistem menggunakan dua komponen utama untuk rekomendasi:

### 5.1. Model Hybrid (Collaborative Filtering + Content-Based) - Deep Learning

* **Kelas Model**: `HybridRecommenderNet(keras.Model)`
* **Arsitektur**:
    * **Input**: Tensor dengan shape `(batch_size, 3 + num_additional_features)`.
        * Kolom 0: `user_encoded` (integer)
        * Kolom 1: `olahraga_encoded` (integer)
        * Kolom 2: `latihan_encoded` (integer)
        * Kolom 3 dst.: Fitur tambahan (`durasi_normal` + fitur pengguna yang sudah di-One-Hot-Encoded).
    * **Embedding Layers**:
        * `user_embedding`: `layers.Embedding(num_users, embedding_size)`
        * `user_bias`: `layers.Embedding(num_users, 1)`
        * `olahraga_embedding`: `layers.Embedding(num_olahraga, embedding_size)`
        * `olahraga_bias`: `layers.Embedding(num_olahraga, 1)`
        * `latihan_embedding`: `layers.Embedding(num_latihan, embedding_size)` (sebagai fitur konten/item)
        * `embedding_size` diset ke `50`.
        * Menggunakan `he_normal` initializer dan `l2(1e-6)` regularizer.
    * **Pemrosesan**:
        1.  Vektor embedding pengguna, olahraga, dan latihan diambil berdasarkan input index.
        2.  Vektor-vektor embedding ini digabungkan (`tf.concat`) dengan fitur tambahan (`additional_feats`).
        3.  Hasil gabungan dilewatkan ke `layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-6))`.
        4.  Output dari dense layer ditambahkan dengan `user_bias` dan `olahraga_bias`.
        5.  Layer output `layers.Dense(1)` menghasilkan prediksi skor.
        6.  Fungsi aktivasi `tf.nn.sigmoid` diterapkan pada output akhir karena target (`kalori_terbakar_normal`) sudah dinormalisasi ke rentang [0,1].
* **Variabel Penting untuk Model**:
    * `num_users`: Jumlah pengguna unik.
    * `num_olahraga`: Jumlah olahraga unik (berdasarkan ID).
    * `num_latihan`: Jumlah jenis latihan unik (berdasarkan nama).
    * `num_add_feats`: Jumlah fitur tambahan (`durasi_normal` + jumlah kolom OHE).

### 5.2. Matriks Similaritas Content-Based (TF-IDF & Cosine Similarity)

* **Fungsi**: `build_content_matrix(df_olahraga)`
* **Proses**:
    1.  Fitur konten dibuat dengan menggabungkan `latihan` (nama) dan `kalori_jam` (diubah jadi string) dari `df_olahraga`.
    2.  `TfidfVectorizer` dari `sklearn.feature_extraction.text` digunakan untuk mengubah fitur konten gabungan menjadi matriks TF-IDF.
    3.  `cosine_similarity` dari `sklearn.metrics.pairwise` dihitung pada matriks TF-IDF untuk mendapatkan matriks similaritas kosinus antar olahraga.
    4.  Hasilnya adalah DataFrame `cb_similarity_df` dengan `id_olahraga` sebagai index dan kolom, berisi skor similaritas antar semua pasangan olahraga.

---

## 6. Pelatihan Model Hybrid

* Model `HybridRecommenderNet` dikompilasi dengan:
    * **Loss Function**: `tf.keras.losses.MeanSquaredError()` (karena target adalah nilai kontinu yang ternormalisasi).
    * **Optimizer**: `keras.optimizers.Adam(learning_rate=0.001)`.
    * **Metrics**: `tf.keras.metrics.MeanSquaredError(name='mse')`.
* Pelatihan dilakukan menggunakan metode `.fit()`:
    * `x`: `X_train`
    * `y`: `y_train`
    * `batch_size`: 8
    * `epochs`: 20 (dapat disesuaikan)
    * `validation_data`: `(X_val, y_val)`
    * `callbacks`: `tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)` untuk mencegah overfitting dan menyimpan bobot terbaik.
* Kurva MSE (Train vs. Validation) divisualisasikan menggunakan `matplotlib.pyplot` untuk memantau performa model.

---

## 7. Proses Rekomendasi Hybrid

* **Fungsi**: `recommend_hybrid_for_user(user_id, df_olahraga, df_history_original, model_hybrid, cb_matrix, metadata, top_k, alpha)`
* **Langkah-langkah**:
    1.  Pastikan `user_id` ada dalam `metadata`.
    2.  Dapatkan `user_encoded` dari `metadata`.
    3.  Identifikasi `visited_items` (olahraga yang pernah dilakukan user) dari `df_history_original`.
    4.  Buat daftar `candidates` (olahraga yang belum pernah dicoba user).
    5.  **Prediksi Skor CF**:
        * Untuk setiap kandidat, buat vektor input `X_cand` dengan `user_encoded`, `olahraga_encoded` kandidat. Fitur `latihan_encoded` dan fitur tambahan (durasi normal + OHE) diasumsikan nol atau nilai default/rata-rata (dalam implementasi ini dibiarkan 0 untuk `additional_feats` selain `user_encoded` dan `olahraga_encoded` saat prediksi untuk kandidat, yang mungkin perlu ditinjau untuk akurasi optimal).
        * Lakukan prediksi menggunakan `model_hybrid.predict(X_cand)` untuk mendapatkan skor CF.
    6.  **Normalisasi Skor CF**: Skor CF dinormalisasi ke rentang [0,1].
    7.  **Hitung Skor CBF**:
        * Untuk setiap kandidat, hitung rata-rata cosine similarity antara kandidat tersebut dengan semua `visited_items` menggunakan `cb_matrix`.
        * Jika tidak ada `visited_items`, skor CBF adalah 0.
    8.  **Normalisasi Skor CBF**: Skor CBF dinormalisasi ke rentang [0,1].
    9.  **Gabungkan Skor**:
        * `hybrid_scores = alpha * cf_norm + (1 - alpha) * cb_norm`
        * `alpha` (default: 0.6) adalah bobot untuk skor CF.
    10. **Pilih Top-K**: Ambil `top_k` (default: 10) olahraga dengan skor hybrid tertinggi.
    11. **Format Hasil**: Kumpulkan hasil rekomendasi dalam DataFrame yang berisi `id_olahraga`, `latihan`, `rata_durasi_menit`, `rata_kalori_terbakar`, dan `skor_hybrid`.

---

## 8. Penyimpanan Model ke TensorFlow.js

* Model Keras (`model_hybrid`) yang telah dilatih disimpan ke format TensorFlow.js (TFJS) menggunakan `tfjs.converters.save_keras_model()`.
* Model disimpan ke direktori `/content/hybrid_recommender_tfjs`.
* Direktori ini kemudian di-zip menjadi `hybrid_recommender_tfjs.zip` dan ditawarkan untuk diunduh. Ini memungkinkan model untuk di-deploy pada aplikasi berbasis web/JavaScript.

---

## 9. Struktur Kode dan Fungsi Utama

Notebook diorganisir ke dalam beberapa bagian utama dengan fungsi-fungsi spesifik:
* **Import Library**: Mengimpor semua pustaka yang dibutuhkan.
* **Data Loading**: Fungsi `load_datasets()`.
* **Exploratory Data Analysis**: Fungsi `exploratory_data()`.
* **Data Preprocessing**: Fungsi-fungsi `merge_data()`, `compute_calories()`, `encode_ids()`, `encode_latihan()`, `one_hot_encode_features()`, `normalize_features()`, `build_X_y()`, `split_train_val()`.
* **Deklarasi Model Hybrid**: Kelas `HybridRecommenderNet`.
* **Training Model Hybrid**: Proses kompilasi dan pelatihan model.
* **Membangun Content Based Matrix**: Fungsi `build_content_matrix()`.
* **Rekomendasi Hybrid**: Fungsi `recommend_hybrid_for_user()`.
* **Penggunaan Fungsionalitas Rekomendasi**: Contoh pemanggilan fungsi rekomendasi.
* **Saved Model to TFJS**: Proses ekspor model.

---

## 10. Ketergantungan (Dependencies)

Pustaka Python utama yang digunakan:
* `pandas`: Untuk manipulasi data.
* `numpy`: Untuk operasi numerik.
* `tensorflow` & `keras`: Untuk membangun dan melatih model Deep Learning.
* `tensorflowjs`: Untuk mengkonversi model Keras ke format TFJS.
* `scikit-learn`: Untuk pra-pemrosesan data (OneHotEncoder, TfidfVectorizer, cosine_similarity).
* `matplotlib`: Untuk visualisasi.
* `google.colab.files`: Untuk fungsionalitas unduh di Google Colab.

Versi spesifik mungkin perlu diperhatikan untuk reproduksibilitas, meskipun notebook ini menggunakan `Requirement already satisfied` untuk `tensorflowjs` yang mengindikasikan versi yang kompatibel sudah terinstal di lingkungan Colab saat eksekusi.

---

## 11. Panduan Setup (Singkat)

1.  **Lingkungan**: Proyek ini dikembangkan dan diuji dalam lingkungan Google Colaboratory.
2.  **Instalasi Pustaka**:
    * Pustaka standar seperti `pandas`, `numpy`, `sklearn`, `matplotlib`, `tensorflow` biasanya sudah tersedia di Colab.
    * Pustaka `tensorflowjs` diinstal secara eksplisit menggunakan:
        ```bash
        !pip install tensorflowjs
        ```
3.  **Menjalankan Notebook**: Eksekusi sel-sel dalam notebook secara berurutan. Data akan dimuat secara otomatis dari GitHub.

---
