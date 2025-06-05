# Ini Alurnya

### 1. **Preprocessing (di file folder utils)**

Langkah pertama adalah membersihkan dan mempersiapkan data masukan:

* **Pemrosesan Teks:** Ini dilakukan di metode `preprocess_text` pada `EnhancedJobAnalyzer` di `analyzer.py`. Teks diubah menjadi huruf kecil, menghapus karakter non-alfanumerik, dan memproses teks menjadi format yang bersih.
* **Ekstraksi dan Pencocokan Keterampilan:** Di `skills_utils.py`, keterampilan yang disebutkan dalam CV dan deskripsi pekerjaan dibandingkan. Keterampilan yang cocok diberi skor berdasarkan keberadaannya, dan persentase kecocokan dihitung.
* **Pencocokan Pengalaman dan Pendidikan:** Fungsi seperti `analyze_experience_match` dan `analyze_education_match` di `experience_utils.py` menilai relevansi antara CV dan pekerjaan berdasarkan jumlah tahun pengalaman dan tingkat pendidikan.
* **Pencocokan Industri:** Fungsi `analyze_industry_match` memeriksa apakah kata kunci industri atau peran yang ada di CV cocok dengan deskripsi pekerjaan.

### 2. **Perhitungan Similarity (di file `similarity_utils.py`)**

Setelah teks diproses, **similarity** antara CV dan deskripsi pekerjaan dihitung. Ini dilakukan dengan:

* **Vektorisasi TF-IDF:** Data teks dari CV dan deskripsi pekerjaan diubah menjadi vektor numerik menggunakan `TfidfVectorizer`. Vektorisasi ini menangkap relevansi kata-kata dalam kedua teks tersebut.
* **Cosine Similarity:** Setelah vektorisasi, **cosine similarity** dihitung untuk mengukur seberapa mirip dua teks tersebut. Cosine similarity memiliki nilai antara 0 hingga 1, di mana 1 menunjukkan kesamaan penuh, dan 0 menunjukkan tidak ada kesamaan.

### 3. **Neural Network (di file `neural_model.py`)**


* **Arsitektur Model:** Model ini menggunakan layer embedding untuk mengonversi kata menjadi vektor, dan menggunakan **Bidirectional LSTM** (Long Short-Term Memory) yang memproses urutan kata dalam teks. Layer **GlobalMaxPooling** mengurangi dimensi output dari LSTM. Setelah itu, digunakan layer **Dense** untuk menghasilkan prediksi.
* **Fitur Numerik:** Selain teks, fitur numerik juga digabungkan, yang dihasilkan dari kata-kata umum antara CV dan job deskripsi. Fitur-fitur numerik ini diproses menggunakan **StandardScaler**.
* **Pelatihan Model:** Model neural network dilatih menggunakan data pelatihan sintetis. Data sintetis ini dibuat dengan memasangkan sampel CV dengan job deskripsi dan menghitung skor kesamaan. Jika skor kesamaan melebihi ambang batas, data ini diberi label cocok.
* **Prediksi:** Setelah dilatih, model dapat memprediksi skor kecocokan antara CV dan deskripsi pekerjaan yang baru.

### 4. **Rekomendasi Pekerjaan (di file `main.py` dan `similarity_utils.py`)**

Setelah CV dan deskripsi pekerjaan diproses dan dianalisis:

* **Feedback detail:** Di `main.py`, setelah prediksi kecocokan dilakukan oleh neural network, **feedback detail** dihasilkan. Laporan ini mencakup berbagai faktor seperti similarity teks, pencocokan skill, pencocokan experience, education, dan industri.
* **Rekomendasi job:** Fungsi `find_alternative_jobs_for_cv` menyarankan pekerjaan alternatif yang cocok dengan CV.

### 5. **Alur Eksekusi di `main.py`**

* **Langkah 1: Upload CV**

  * Fungsi `upload_cv()` di `pdf_utils.py` menangani pengunggahan file PDF, mengekstraksi teks dari CV, dan memprosesnya.

* **Langkah 2: Input Job Deskripsi**

  * User memasukkan judul pekerjaan dan job deskripsi.
  * **Pemeriksaan bahasa** memastikan job deskripsi menggunakan bahasa Inggris dengan menggunakan `validate_english_text` dari `language_utils.py`.

* **Langkah 3: Load Data Pekerjaan**

  * Dataset (`fake_job_postings.csv`) diload, yang berisi postingan pekerjaan.

* **Langkah 4: Melatih Neural Network**

  * Data pelatihan sintetis dibuat, dan model neural network dilatih menggunakan `train_neural_network()`.

* **Langkah 5: Menghasilkan Feedback Detail**

  * Setelah pelatihan, sistem menghasilkan feedback yang mendetail yang membandingkan CV dengan deskripsi pekerjaan, termasuk similarity dan skor kecocokan.

* **Langkah 6: Rekomendasi Pekerjaan**

  * Berdasarkan analisis, sistem menyarankan pekerjaan alternatif yang cocok dengan CV.

### 6. **Alur Latihan dan Prediksi Model**

* **Membuat Data Pelatihan:** Di `create_synthetic_training_data`, data pelatihan dibuat dengan memasangkan teks CV dengan deskripsi pekerjaan dan menghitung skor kesamaan.
* **Pelatihan:** Neural network dilatih dengan data sintetis ini untuk memprediksi kecocokan pekerjaan.
* **Prediksi:** Model kemudian memprediksi kecocokan untuk CV dan deskripsi pekerjaan yang tidak terlihat sebelumnya menggunakan model yang telah dilatih.

### 7. **Sebagian besar library yang Digunakan**

* **PyPDF2:** Digunakan untuk ekstraksi teks dari PDF.
* **Langdetect:** Untuk mendeteksi bahasa dalam deskripsi pekerjaan dan CV.
* **Scikit-learn:** Untuk vektorisasi TF-IDF, cosine similarity, dan tugas pembelajaran mesin lainnya.
* **TensorFlow/Keras:** Untuk membangun dan melatih model pembelajaran mendalam.

---

### Ringkasan

1. **Preprocessing:** Membersihkan dan mengekstraksi fitur dari CV dan deskripsi pekerjaan.
2. **Perhitungan Similarity:** Menggunakan TF-IDF dan cosine similarity untuk membandingkan teks.
3. **Prediksi Neural Network:** Model neural betwork yang memprediksi kecocokan berdasarkan data pelatihan.
4. **Rekomendasi Pekerjaan:** Memberikan pekerjaan serupa /relevan berdasarkan cv yang sebelumnya diunggah oleh user,
