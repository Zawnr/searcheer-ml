**Version**: 2.2.0  
**Base URL**: `http://localhost:8000`  
**Content-Type**: `application/json` (untuk JSON endpoints), `multipart/form-data` (untuk file upload)

## Rate Limiting

| Endpoint | Limit |
|----------|-------|
| Default | 100 requests/hour |
| `/api/cv/upload` | 10 requests/minute |
| `/api/analyze/cv-with-job` | 10 requests/minute |
| `/api/find-alternative-jobs` | 10 requests/minute |

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

## Endpoints

### 1. Root Endpoint

**GET** `/`

Menampilkan informasi dasar API dan daftar endpoint yang tersedia.

**Response:**
```json
{
  "message": "Job Compatibility Analyzer REST API v2.2.0",
  "documentation": "/api/docs",
  "endpoints": {
    "health": "/api/health",
    "upload_cv": "/api/cv/upload",
    "analyze_compatibility": "/api/analyze/compatibility",
    "analyze_cv_with_job": "/api/analyze/cv-with-job",
    "analyze_flexible": "/api/analyze/flexible"
  }
}
```

### 2. Health Check

**GET** `/api/health`

Memeriksa status kesehatan API dan komponen-komponennya.

**Response:**
```json
{
  "success": true,
  "message": "Service is healthy",
  "data": {
    "status": "healthy",
    "analyzer_ready": true,
    "dataset_loaded": true,
    "dataset_size": 15000,
    "timestamp": "2024-01-15T10:30:00.000Z"
  },
  "errors": [],
  "timestamp": "2024-01-15T10:30:00.000Z",
  "request_id": "uuid-here"
}
```

### 3. Upload CV

**POST** `/api/cv/upload`

Upload file CV (PDF) dan ekstraksi teks dengan validasi ATS-friendly.

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): File PDF CV

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/cv/upload \
  -F "file=@/path/to/cv.pdf"
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "CV processed successfully",
  "data": {
    "cv_text": "Extracted CV content...",
    "ats_score": 85.5,
    "word_count": 450,
    "language_detected": "en",
    "readability_score": 0.855
  },
  "errors": [],
  "timestamp": "2024-01-15T10:30:00.000Z",
  "request_id": "uuid-here"
}
```

**Error Response (400) - ATS Issues:**
```json
{
  "success": false,
  "message": "CV needs improvement for ATS compatibility",
  "data": {
    "ats_score": 45.0,
    "ats_issues": [
      "Missing essential sections (Contact, Experience, Education, Skills)",
      "No valid email address found"
    ],
    "word_count": 120,
    "cv_preview": "CV content preview..."
  },
  "errors": [
    "Missing essential sections (Contact, Experience, Education, Skills)",
    "No valid email address found"
  ]
}
```

### 4. Analyze CV with Job

**POST** `/api/analyze/cv-with-job`

Menganalisis kecocokan file CV dengan job description.

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): File PDF CV
- `job_title` (required): Judul pekerjaan (min 3 karakter)
- `job_description` (required): Deskripsi pekerjaan (min 5 kata)

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/analyze/cv-with-job \
  -F "file=@/path/to/cv.pdf" \
  -F "job_title=Senior Data Scientist" \
  -F "job_description=We are looking for an experienced data scientist with Python, SQL, machine learning expertise..."
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "CV analysis and job compatibility completed successfully",
  "data": {
    "cv_analysis": {
      "ats_score": 85.5,
      "word_count": 450,
      "language_detected": "en"
    },
    "compatibility_analysis": {
      "overall_score": 78.5,
      "text_similarity": 65.2,
      "skill_match": 82.0,
      "experience_match": 75.0,
      "education_match": 80.0,
      "industry_match": 85.0,
      "recommendation_level": "STRONG_MATCH",
      "matched_skills": ["python", "sql", "machine learning"],
      "missing_skills": ["tensorflow", "aws"],
      "tips": [
        "Highlight your matching skills prominently",
        "Prepare specific examples of relevant experience",
        "Apply with confidence",
        "Research the company culture and values"
      ],
      "confidence_score": 0.78
    }
  },
  "errors": [],
  "timestamp": "2024-01-15T10:30:00.000Z",
  "request_id": "uuid-here"
}
```

### 5. Find Alternative Jobs

**POST** `/api/find-alternative-jobs`

Mencari rekomendasi pekerjaan berdasarkan hasil analisis CV.

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "cv_text": "CV content here...",
  "analysis_results": {
    "overall_score": 78.5,
    "skills_analysis": {
      "matched_skills": [["python", 1.0], ["sql", 1.0]],
      "missing_skills": [["tensorflow", 1.0]],
      "skill_match_percentage": 82.0
    }
  },
  "top_n": 5
}
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Alternative job recommendations found",
  "data": {
    "recommended_jobs": [
      {
        "title": "Machine Learning Engineer",
        "company": "Tech Corp",
        "description": "Job description...",
        "compatibility_score": 85.2,
        "matched_skills": ["python", "sql", "machine learning"],
        "location": "Remote"
      },
      {
        "title": "Data Analyst",
        "company": "Analytics Inc",
        "description": "Job description...",
        "compatibility_score": 79.8,
        "matched_skills": ["python", "sql"],
        "location": "New York"
      }
    ]
  },
  "errors": [],
  "timestamp": "2024-01-15T10:30:00.000Z",
  "request_id": "uuid-here"
}
```

### 6. API Documentation

**GET** `/api/docs`

Menampilkan dokumentasi API.

**Response:**
```json
{
  "title": "Job Compatibility Analyzer REST API",
  "version": "2.2.0",
  "description": "REST API for analyzing CV compatibility with job descriptions",
  "endpoints": {
    "GET /": "Root endpoint with basic info",
    "GET /api/health": "Health check endpoint",
    "POST /api/cv/upload": "Upload and process CV file",
    "POST /api/analyze/compatibility": "Analyze compatibility (JSON input)",
    "POST /api/analyze/cv-with-job": "Analyze CV file with job details (Form data)",
    "POST /api/analyze/flexible": "Flexible analysis (file or text input)"
  },
  "rate_limits": {
    "default": "100 requests per hour",
    "upload_cv": "10 requests per minute",
    "analyze_compatibility": "30 requests per minute",
    "analyze_cv_with_job": "10 requests per minute",
    "analyze_flexible": "10 requests per minute"
  }
}
```

## Error Codes

| HTTP Code | Description | Common Causes |
|-----------|-------------|---------------|
| 400 | Bad Request | Invalid input, missing required fields, file validation failed |
| 404 | Not Found | Endpoint tidak ditemukan |
| 405 | Method Not Allowed | HTTP method tidak didukung |
| 413 | Payload Too Large | File size melebihi 10MB |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error, analyzer not ready |

## Error Response Examples

**400 - Validation Error:**
```json
{
  "success": false,
  "message": "Invalid job title",
  "data": null,
  "errors": ["Job title must be at least 3 characters"],
  "timestamp": "2024-01-15T10:30:00.000Z",
  "request_id": "uuid-here"
}
```

**429 - Rate Limit:**
```json
{
  "success": false,
  "message": "Rate limit exceeded",
  "data": null,
  "errors": ["Too many requests. Please try again later."],
  "timestamp": "2024-01-15T10:30:00.000Z",
  "request_id": "uuid-here"
}
```

## ATS Compatibility Scoring


| Kriteria | Bobot | Deskripsi |
|----------|-------|-----------|
| Essential Sections | 30 poin | Contact, Experience, Education, Skills |
| Contact Information | 20 poin | Valid email dan phone number |
| Content Length | 15 poin | Minimum 200 kata |
| Structure Indicators | 15 poin | Dates, degrees, professional terms |
| Professional Keywords | 10 poin | Action words, experience descriptions |
| Character Validation | 10 poin | Minimal special characters |

**ATS Score ≥ 70 + Issues ≤ 2 = ATS Compatible**

## File Requirements

### CV Upload Requirements
- **Format**: PDF only
- **Size**: Maximum 10MB
- **Language**: Harus berbahasa inggris
- **Content**: Harus informatif (minimum 100 kata)
- **Structure**: Harus memenuhi format standard
