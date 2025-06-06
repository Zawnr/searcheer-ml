from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import pandas as pd
import os
import uvicorn
import logging
import uuid
import time
import json
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
import re
from functools import wraps

class Settings(BaseSettings):
    dataset_path: str = "data/fake_job_postings.csv"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = [".pdf"]
    max_requests_per_minute: int = 30
    enable_neural_network: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()

#konfigurasi logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

analyzer = None
job_data = None
request_counts = {}

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = None

def clean_text_safe(text: Any) -> str:
    try:
        #mengonversi ke string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        if not isinstance(text, str):
            text = str(text)
        
        #membersihkan karakter yang tidak dapat dicetak
        cleaned = ''.join(char for char in text if char.isprintable() or char in '\n\r\t ')
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    except Exception:
        return ""

class JobAnalysisRequest(BaseModel):
    job_title: str
    job_description: str
    cv_text: str

class EnhancedCompatibilityResponse(BaseModel):
    overall_score: float
    text_similarity: float
    skill_match: float
    experience_match: float
    education_match: float
    industry_match: float
    recommendation_level: str
    matched_skills: List[str]
    missing_skills: List[str]
    tips: List[str]
    confidence_score: float

class EnhancedCVUploadResponse(BaseModel):
    cv_text: Optional[str] = None
    ats_score: Optional[float] = None
    ats_issues: Optional[List[str]] = None
    word_count: Optional[int] = None
    language_detected: Optional[str] = None
    readability_score: Optional[float] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer, job_data
    logger.info("Starting Job Compatibility API...")
    
    try:
        from analyzer import EnhancedJobAnalyzer
        analyzer = EnhancedJobAnalyzer()
        logger.info("Analyzer initialized")
        
        if os.path.exists(settings.dataset_path):
            job_data = pd.read_csv(settings.dataset_path)
            job_data = job_data[job_data.get("fraudulent", 0) == 0]
            job_data = job_data.dropna(subset=['title', 'description'])
            logger.info(f"Dataset loaded: {len(job_data)} jobs")
            
            if settings.enable_neural_network:
                asyncio.create_task(train_analyzer_background())
        else:
            logger.warning(f"Dataset not found at {settings.dataset_path}")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    logger.info("Shutting down Job Compatibility API...")

def rate_limit(max_requests: int = 30, window_seconds: int = 60):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            current_time = time.time()
            
            #cleanup old requests
            for ip in list(request_counts.keys()):
                request_counts[ip] = [
                    timestamp for timestamp in request_counts[ip] 
                    if current_time - timestamp < window_seconds
                ]
                if not request_counts[ip]:
                    del request_counts[ip]
            
            if client_ip not in request_counts:
                request_counts[client_ip] = []
            
            if len(request_counts[client_ip]) >= max_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )
            
            request_counts[client_ip].append(current_time)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

class APIException(HTTPException):
    def __init__(self, status_code: int, message: str, details: Dict = None):
        super().__init__(
            status_code=status_code,
            detail={
                "message": message,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
        )

async def validate_uploaded_file(file: UploadFile) -> bytes:
    if not file.filename.lower().endswith('.pdf'):
        raise APIException(400, "Only PDF files are supported")
    
    content = await file.read()
    
    if len(content) > settings.max_file_size:
        raise APIException(400, f"File too large. Maximum size: {settings.max_file_size // (1024*1024)}MB")
    
    if not content.startswith(b'%PDF'):
        raise APIException(400, "Invalid PDF file format")
    
    return content

async def train_analyzer_background():
    try:
        global analyzer, job_data
        if analyzer and job_data is not None:
            logger.info("Training neural network...")
            analyzer.train_neural_network(job_data)
            logger.info("Neural network training completed")
    except Exception as e:
        logger.error(f"Background training error: {e}")

def perform_analysis(cv_text: str, job_title: str, job_description: str):
    """Fungsi helper untuk melakukan analisis kompatibilitas"""
    if not analyzer:
        raise APIException(503, "Analyzer service unavailable")
    
    #mengecek bahasa
    from utils.language_utils import validate_english_text
    job_lang_check = validate_english_text(job_description, "Job Description")
    if not job_lang_check['is_english']:
        raise APIException(400, "Job description must be in English")
    
    #generate analysis
    from utils.similarity_utils import generate_detailed_analysis_report
    results = generate_detailed_analysis_report(
        cv_text,
        job_title,
        job_description,
        analyzer
    )
    
    if not results:
        raise APIException(500, "Analysis computation failed")
    
    #process hasil analisis
    skills_analysis = results.get('skills_analysis', {})
    matched_skills = [skill for skill, _ in skills_analysis.get('matched_skills', [])]
    missing_skills = [skill for skill, _ in skills_analysis.get('missing_skills', [])]
    
    overall_score = results['overall_score']
    confidence_score = min(
        (results['skill_match'] + results['text_similarity']) / 200 + 0.3,
        1.0
    )

    if overall_score < 45:
        recommendation_level = "LOW_MATCH"
        tips = [
            "Focus on developing fundamental skills",
            "Consider entry-level positions or internships",
            "Take relevant courses or certifications",
            "Build a portfolio to demonstrate skills"
        ]
    elif overall_score < 70:
        recommendation_level = "MODERATE_MATCH"
        tips = [
            "Develop the missing high-priority skills",
            "Gain experience through projects or volunteering",
            "Tailor your CV to highlight relevant experience",
            "Consider similar roles to build experience"
        ]
    else:
        recommendation_level = "STRONG_MATCH"
        tips = [
            "Highlight your matching skills prominently",
            "Prepare specific examples of relevant experience",
            "Apply with confidence",
            "Research the company culture and values"
        ]
    
    return {
        'overall_score': overall_score,
        'text_similarity': results['text_similarity'],
        'skill_match': results['skill_match'],
        'experience_match': results['experience_match'],
        'education_match': results['education_match'],
        'industry_match': results['industry_match'],
        'recommendation_level': recommendation_level,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'tips': tips,
        'confidence_score': confidence_score
    }

#inisialisasi FastAPI 
app = FastAPI(
    title="Job Compatibility Analyzer API",
    description="Advanced API for analyzing CV compatibility with job descriptions",
    version="2.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

#konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#request middleware untuk logging
@app.middleware("http")
async def process_request(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed in {process_time:.2f}s")
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = str(process_time)
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed after {process_time:.2f}s: {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "errors": ["Request processing failed"],
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
        )

#error handler untuk validasi request
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    logger.error(f"Validation error for request {request_id}: {exc}")
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Request validation failed",
            "errors": [str(exc)],
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
    )

@app.get("/")
async def root():
    return {"message": "Job Compatibility Analyzer API v2.2.0", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return StandardResponse(
        success=True,
        message="Service is healthy",
        data={
            "status": "healthy",
            "analyzer_ready": analyzer is not None,
            "dataset_loaded": job_data is not None,
            "dataset_size": len(job_data) if job_data is not None else 0
        }
    )

@app.post("/upload-cv")
@rate_limit(max_requests=10, window_seconds=60)
async def upload_cv(request: Request, file: UploadFile = File(...)):
    """Mengupload CV yang nanti outputnya berupa teks yang berhasil diekstrak dri pdf"""
    try:
        file_content = await validate_uploaded_file(file)
        
        from utils.pdf_utils import extract_text_from_pdf, is_ats_friendly
        from utils.language_utils import validate_english_text
        
        cv_text = extract_text_from_pdf(file_content)
        cv_text = clean_text_safe(cv_text)
        word_count = len(cv_text.split())
        
        if word_count < 100:
            return StandardResponse(
                success=False,
                message="CV content insufficient",
                errors=["CV must contain at least 100 words for proper analysis"],
                data={"word_count": word_count},
                request_id=request.state.request_id
            )
        
        language_result = validate_english_text(cv_text, text_type="CV")
        if not language_result['is_english']:
            return StandardResponse(
                success=False,
                message="Language validation failed",
                errors=[language_result['message']],
                request_id=request.state.request_id
            )
        
        ats_compatible, ats_issues, ats_score = is_ats_friendly(cv_text)
        
        return StandardResponse(
            success=ats_compatible,
            message="CV processed successfully" if ats_compatible else "CV needs improvement",
            data=EnhancedCVUploadResponse(
                cv_text=cv_text if ats_compatible else None,
                ats_score=float(ats_score),
                ats_issues=ats_issues,
                word_count=word_count,
                language_detected=language_result['detected_language'],
                readability_score=min(ats_score / 100, 1.0)
            ).dict(),
            errors=ats_issues if not ats_compatible else None,
            request_id=request.state.request_id
        )
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"CV upload error: {e}")
        raise APIException(500, f"CV processing failed: {str(e)}")

@app.post("/analyze-cv-with-job")
@rate_limit(max_requests=10, window_seconds=60)
async def analyze_cv_with_job(
    request: Request,
    file: UploadFile = File(...),
    job_title: str = Form(...),
    job_description: str = Form(...)
):
    try:
        #membersihkan data input
        job_title = clean_text_safe(job_title)
        job_description = clean_text_safe(job_description)
        
        #memeriksa data input
        if len(job_title) < 5:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "Job title too short",
                    "errors": ["Job title must be at least 5 characters"],
                    "timestamp": datetime.now().isoformat(),
                    "request_id": getattr(request.state, 'request_id', str(uuid.uuid4()))
                }
            )
        
        if len(job_description.split()) < 10:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "Job description too short", 
                    "errors": ["Job description must be at least 10 words"],
                    "timestamp": datetime.now().isoformat(),
                    "request_id": getattr(request.state, 'request_id', str(uuid.uuid4()))
                }
            )
        
        file_content = await validate_uploaded_file(file)
        
        from utils.pdf_utils import extract_text_from_pdf, is_ats_friendly
        from utils.language_utils import validate_english_text
        
        cv_text = extract_text_from_pdf(file_content)
        cv_text = clean_text_safe(cv_text)
        word_count = len(cv_text.split())
        
        if word_count < 100:
            return StandardResponse(
                success=False,
                message="CV content insufficient",
                errors=["CV must contain at least 100 words for proper analysis"],
                data={"word_count": word_count},
                request_id=request.state.request_id
            )
        
        #deteksi bahasa CV
        language_result = validate_english_text(cv_text, text_type="CV")
        if not language_result['is_english']:
            return StandardResponse(
                success=False,
                message="CV language validation failed",
                errors=[language_result['message']],
                request_id=request.state.request_id
            )
        
        #deteksi bahasa job description
        job_lang_check = validate_english_text(job_description, "Job Description")
        if not job_lang_check['is_english']:
            return StandardResponse(
                success=False,
                message="Job description language validation failed",
                errors=["Job description must be in English"],
                request_id=request.state.request_id
            )
        
        #mengecek apakah CV ats friendly
        ats_compatible, ats_issues, ats_score = is_ats_friendly(cv_text)
        
        if not ats_compatible:
            return StandardResponse(
                success=False,
                message="CV needs improvement for ATS compatibility",
                errors=ats_issues,
                data={
                    "ats_score": float(ats_score),
                    "word_count": word_count,
                    "cv_preview": cv_text[:200] + "..." if len(cv_text) > 200 else cv_text
                },
                request_id=request.state.request_id
            )
        
        analysis_result = perform_analysis(cv_text, job_title, job_description)
        
        response_data = {
            "cv_analysis": {
                "ats_score": float(ats_score),
                "word_count": word_count,
                "language_detected": language_result['detected_language']
            },
            "compatibility_analysis": analysis_result
        }
        
        return StandardResponse(
            success=True,
            message="CV analysis and job compatibility completed successfully",
            data=response_data,
            request_id=request.state.request_id
        )
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"CV analysis error: {e}")
        raise APIException(500, f"Analysis failed: {str(e)}")

#hanya bisa menerima JSON
@app.post("/analyze-compatibility")
@rate_limit(max_requests=settings.max_requests_per_minute, window_seconds=60)
async def analyze_compatibility(request: Request, data: JobAnalysisRequest):
    try:
        #membersihkan data input
        job_title = clean_text_safe(data.job_title)
        job_description = clean_text_safe(data.job_description)
        cv_text = clean_text_safe(data.cv_text)
        
        #validasi dasar
        if len(job_title) < 5:
            raise APIException(422, "Job title must be at least 5 characters")
        
        if len(job_description.split()) < 10:
            raise APIException(422, "Job description must be at least 10 words")
        
        if len(cv_text.split()) < 100:
            raise APIException(422, "CV text must be at least 100 words")
        
        analysis_result = perform_analysis(cv_text, job_title, job_description)
        
        response_data = EnhancedCompatibilityResponse(**analysis_result)
        
        return StandardResponse(
            success=True,
            message="Analysis completed successfully",
            data=response_data.dict(),
            request_id=request.state.request_id
        )
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise APIException(500, f"Analysis failed: {str(e)}")

#bisa menerima file atau teks
@app.post("/analyze-compatibility-flexible")
@rate_limit(max_requests=10, window_seconds=60)
async def analyze_compatibility_flexible(
    request: Request,
    file: UploadFile = File(None),
    cv_text: str = Form(None),
    job_title: str = Form(...),
    job_description: str = Form(...)
):
    try:
        #membersihkan data input
        job_title = clean_text_safe(job_title)
        job_description = clean_text_safe(job_description)
        
        if len(job_title) < 5:
            raise APIException(422, "Job title must be at least 5 characters")
        
        if len(job_description.split()) < 10:
            raise APIException(422, "Job description must be at least 10 words")
        
        #mengekstrak teks dari file atau menggunakan teks yang diberikan
        if file:
            file_content = await validate_uploaded_file(file)
            from utils.pdf_utils import extract_text_from_pdf, is_ats_friendly
            cv_text = extract_text_from_pdf(file_content)
            cv_text = clean_text_safe(cv_text)
        elif cv_text:
            cv_text = clean_text_safe(cv_text)
        else:
            raise APIException(422, "Either file or cv_text must be provided")
        
        if len(cv_text.split()) < 100:
            raise APIException(422, "CV text must be at least 100 words")
        
        analysis_result = perform_analysis(cv_text, job_title, job_description)
        
        response_data = EnhancedCompatibilityResponse(**analysis_result)
        
        return StandardResponse(
            success=True,
            message="Analysis completed successfully",
            data=response_data.dict(),
            request_id=request.state.request_id
        )
        
    except APIException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise APIException(500, f"Analysis failed: {str(e)}")

# Error handlers
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    return JSONResponse(
        status_code=exc.status_code,
        content=StandardResponse(
            success=False,
            message=exc.detail["message"],
            errors=[exc.detail["message"]],
            data=exc.detail.get("details", {}),
            timestamp=exc.detail["timestamp"],
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=StandardResponse(
            success=False,
            message="An unexpected error occurred",
            errors=["Internal server error"],
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
