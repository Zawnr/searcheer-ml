# Searcheer

**Version**: `2.2.0`
**Base URL**: `http://localhost:8000`
**Content-Type**: `application/json` (untuk JSON endpoints), `multipart/form-data` (untuk file upload)

---

## Cara Menjalankan

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

### 2. Menjalankan Proyek Machine Learning

```bash
python main.py
```

### 3. Menjalankan REST API

```bash
python api.py
```

---

## Contoh Pengujian dengan cURL

### Health Check

```bash
curl http://127.0.0.1:8000/api/health
```

### Upload CV

```bash
curl.exe -X POST http://127.0.0.1:8000/api/cv/upload -F "file=@/path/to/cv.pdf"
```

### Analisis CV vs Job Description

```bash
curl.exe -X POST "http://127.0.0.1:8000/api/analyze/cv-with-job" \
  -F "file=@/path/to/cv.pdf" \
  -F "job_title=Data Scientist" \
  -F "job_description=Deskripsi lengkap pekerjaan..."
```

### Rekomendasi Pekerjaan Alternatif

```bash
curl.exe -X POST http://127.0.0.1:8000/api/find-alternative-jobs \
  -H "Content-Type: application/json" \
  --data-binary "@payload.json"
```

---

## Rate Limiting

| Endpoint                     | Limit             |
| ---------------------------- | ----------------- |
| Semua endpoint (default)     | 100 requests/jam  |
| `/api/cv/upload`             | 10 requests/menit |
| `/api/analyze/cv-with-job`   | 10 requests/menit |
| `/api/find-alternative-jobs` | 10 requests/menit |

---

## Format Standard Response

```json
{
  "success": boolean,
  "message": string,
  "data": object|null,
  "errors": array,
  "timestamp": string,
  "request_id": string
}
```

---

## API Endpoints

### 1. Root

**GET** `/`

Menampilkan info API dan daftar endpoint.

---

### 2. Health Check

**GET** `/api/health`

Cek status API dan model.

**Response:**

```json
{
  "success": true,
  "message": "Service is healthy",
  "data": {
    "status": "healthy",
    "analyzer_ready": true,
    "dataset_loaded": true,
    "dataset_size": 15000
  }
}
```

---

### 3. Upload CV

**POST** `/api/cv/upload`

Ekstraksi teks dari CV PDF dan validasi ATS.

* **Content-Type**: `multipart/form-data`
* **Parameter**: `file` (PDF, max 10MB)

**Success Response:**

```json
{
  "api_version": "2.2.0",
  "success": true,
  "message": "CV processed successfully",
  "data": {
    "ats_compatible": true,
    "ats_score": 78.5,
    "character_count": 7669,
    "word_count": 1015,
    "language_detected": "en",
    "readability_score": 0.785,
    "cv_text": "INSANIA CINDY PUAN FADILAHSARI ...", 
    "file_info": {
      "filename": "Insania_Cindy_Puan_Fadilahsari-resume_2.pdf",
      "size_bytes": 93608,
      "processing_time": 0.8682496547698975
    }
  },
  "errors": [],
  "request_id": "7805cb60-4c32-4c77-986d-0d3e24bdf2d4",
  "timestamp": "2025-06-07T12:27:40.919350Z"
}
```

**Error Response (400 - ATS Issues):**

```json
{
  "success": false,
  "message": "CV needs improvement for ATS compatibility",
  "data": {
    "ats_score": 45.0,
    "ats_issues": [
      "Missing essential sections",
      "No valid email address"
    ]
  }
}
```

---

### 4. Analyze CV with Job

**POST** `/api/analyze/cv-with-job`

Menganalisis kecocokan CV dengan deskripsi pekerjaan.

* **Content-Type**: `multipart/form-data`
* **Parameters**:

  * `file`: CV dalam format PDF
  * `job_title`: Judul pekerjaan
  * `job_description`: Deskripsi pekerjaan

**Response:**

```json
{
  "api_version": "2.2.0",
  "success": true,
  "message": "CV analysis and job compatibility completed successfully",
  "data": {
    "compatibility_analysis": {
      "confidence_score": 0.3,
      "education_match": 50.0,
      "experience_match": 50.0,
      "industry_match": 50.0,
      "skill_match": 80.0,
      "text_similarity": 100.0,
      "overall_score": 100.0,
      "recommendation_level": "MODERATE_MATCH",
      "matched_skills": [],
      "missing_skills": [],
      "analysis_metadata": {
        "common_words_count": 52,
        "fallback_mode": true
      },
      "tips": [
        "Analysis performed with limited capabilities. Consider uploading a more detailed CV."
      ]
    },
    "cv_analysis": {
      "ats_compatible": true,
      "ats_score": 78.5,
      "character_count": 7669,
      "word_count": 1015,
      "language_detected": "en"
    },
    "job_analysis": {
      "title": "Data Scientist",
      "description_length": 2207,
      "description_word_count": 289
    },
    "processing_info": {
      "analysis_time": 0.5449292659759521,
      "analyzer_version": "2.2.0",
      "neural_network_used": true
    }
  },
  "errors": [],
  "request_id": "aecb3dd6-3b6e-49eb-938d-c3c21aeeba6d",
  "timestamp": "2025-06-07T12:25:41.691584Z"
}

```

---

### 5. Find Alternative Jobs

**POST** `/api/find-alternative-jobs`

Merekomendasikan pekerjaan lain yang lebih cocok berdasarkan hasil analisis.

* **Content-Type**: `application/json`

**Request:**

```json
{
  "cv_text": "CV content here...",
  "analysis_results": {
    "overall_score": 78.5,
    "skills_analysis": {
      "matched_skills": [["python", 1.0]],
      "missing_skills": [["tensorflow", 1.0]],
      "skill_match_percentage": 82.0
    }
  },
  "top_n": 5
}
```

**Response:**

```json
{
  "api_version": "2.2.0",
  "data": {
    "metadata": {
      "algorithm_version": "2.2.0",
      "search_time": 162.5912845134735
    },
    "recommended_jobs": [
      {
        "job_id": "2448",
        "job_title": "VP Engineering Operations",
        "description": "Qubit: Cutting Edge Big Data EngineeringQubit's platform collects, stores and processes over 1 billi...",
        "rank": 1,
        "score": 0.5657
      },
      {
        "job_id": "7335",
        "job_title": "Data Scientist",
        "description": "We want to add some fresh talent to our data team to make sure it can fully continue its mission of ...",
        "rank": 2,
        "score": 0.5604
      },
      {
        "job_id": "15375",
        "job_title": "Data Scientist",
        "description": "We want to add some talent to our data team to make sure it can continue its mission of turning the ...",
        "rank": 3,
        "score": 0.5601
      },
      {
        "job_id": "16105",
        "job_title": "Software Engineer - GIS Specialist",
        "description": "Summary: FHU is currently looking to expand our GIS services to include software application develop...",
        "rank": 4,
        "score": 0.5538
      },
      {
        "job_id": "16212",
        "job_title": "Graduate Software Engineer",
        "description": "Qubit: Cutting Edge Big Data EngineeringQubit is opening its third international office in Lahore, P...",
        "rank": 5,
        "score": 0.5454
      }
    ],
    "search_parameters": {
      "cv_word_count": 1015,
      "dataset_size": 17001,
      "top_n": 5
    }
  },
  "errors": [],
  "message": "Found 5 alternative job recommendations",
  "request_id": "11bd204c-907e-43d3-883e-059eede442e6",
  "success": true,
  "timestamp": "2025-06-07T16:26:46.421090Z"
}

```

---

### 6. API Documentation

**GET** `/api/docs`

Menampilkan dokumentasi lengkap API.

---

## Error Handling

| HTTP Code | Deskripsi             | Penyebab Umum                        |
| --------- | --------------------- | ------------------------------------ |
| 400       | Bad Request           | Input tidak valid, file salah format |
| 404       | Not Found             | Endpoint tidak ditemukan             |
| 405       | Method Not Allowed    | HTTP method salah                    |
| 413       | Payload Too Large     | File lebih dari 10MB                 |
| 429       | Too Many Requests     | Melebihi limit                       |
| 500       | Internal Server Error | Kesalahan server / analisis gagal    |

---

## ATS Compatibility Scoring

| Kriteria               | Bobot | Deskripsi                              |
| ---------------------- | ----- | -------------------------------------- |
| Essential Sections     | 30    | Contact, Experience, Education, Skills |
| Contact Info           | 20    | Email dan nomor telepon valid          |
| Panjang Konten         | 15    | Minimal 200 kata                       |
| Struktur Profesional   | 15    | Tanggal, gelar, istilah teknis         |
| Kata Kunci Profesional | 10    | Kata kerja, kata spesifik industri     |
| Validasi Karakter      | 10    | Minim karakter non-standar             |

**Note**:

* **ATS Score ≥ 70** dan **≤ 2 masalah** dianggap **ATS Compatible**

---

## File Requirements

### CV Upload

* **Format**: PDF
* **Ukuran Maks**: 10MB
* **Bahasa**: Inggris
* **Konten**: Minimal 100 kata
* **Struktur**: Harus mengandung section penting

---
