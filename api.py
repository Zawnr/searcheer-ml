from fastapi import FastAPI, UploadFile, File, HTTPException, Request
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

# Configure logging
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
    """Ultra-safe text cleaning function"""
    try:
        # Convert to string first
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        
        if not isinstance(text, str):
            text = str(text)
        
        # Remove all non-printable characters except basic whitespace
        cleaned = ''.join(char for char in text if char.isprintable() or char in '\n\r\t ')
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    except Exception:
        return ""

# Simple request models without complex validation
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

# Initialize FastAPI app
app = FastAPI(
    title="Job Compatibility Analyzer API",
    description="Advanced API for analyzing CV compatibility with job descriptions",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple request processing middleware
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

# Simple validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    logger.error(f"Validation error for request {request_id}: {exc}")
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Request validation failed",
            "errors": ["Invalid request format - please check your input"],
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
    )

@app.post("/analyze-cv-with-job")
@rate_limit(max_requests=10, window_seconds=60)
async def analyze_cv_with_job(
    request: Request,
    file: UploadFile = File(...),
    job_title: str = None,
    job_description: str = None
):
    """Upload CV file and analyze compatibility with job in one step"""
    try:
        # Validate job data from form
        if not job_title or not job_description:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "Missing job information",
                    "errors": ["Both job_title and job_description are required"],
                    "timestamp": datetime.now().isoformat(),
                    "request_id": getattr(request.state, 'request_id', str(uuid.uuid4()))
                }
            )
        
        # Clean job data
        job_title = clean_text_safe(job_title)
        job_description = clean_text_safe(job_description)
        
        # Validate job data
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
        
        # Process CV file
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
        
        # Language validation
        language_result = validate_english_text(cv_text, text_type="CV")
        if not language_result['is_english']:
            return StandardResponse(
                success=False,
                message="CV language validation failed",
                errors=[language_result['message']],
                request_id=request.state.request_id
            )
        
        # Job description language validation
        job_lang_check = validate_english_text(job_description, "Job Description")
        if not job_lang_check['is_english']:
            return StandardResponse(
                success=False,
                message="Job description language validation failed",
                errors=["Job description must be in English"],
                request_id=request.state.request_id
            )
        
        # ATS compatibility check
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
        
        # Perform compatibility analysis
        if not analyzer:
            raise APIException(503, "Analyzer service unavailable")
        
        from utils.similarity_utils import generate_detailed_analysis_report
        results = generate_detailed_analysis_report(
            cv_text,
            job_title,
            job_description,
            analyzer
        )
        
        if not results:
            raise APIException(500, "Analysis computation failed")
        
        # Process results
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
        
        response_data = {
            "cv_analysis": {
                "ats_score": float(ats_score),
                "word_count": word_count,
                "language_detected": language_result['detected_language']
            },
            "compatibility_analysis": {
                "overall_score": overall_score,
                "text_similarity": results['text_similarity'],
                "skill_match": results['skill_match'],
                "experience_match": results['experience_match'],
                "education_match": results['education_match'],
                "industry_match": results['industry_match'],
                "recommendation_level": recommendation_level,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "tips": tips,
                "confidence_score": confidence_score
            }
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

@app.get("/")
async def root():
    return {"message": "Job Compatibility Analyzer API v2.1.0", "docs": "/docs"}

@app.post("/upload-cv")
@rate_limit(max_requests=10, window_seconds=60)
async def upload_cv(request: Request, file: UploadFile = File(...)):
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

@app.post("/analyze-compatibility-simple")
@rate_limit(max_requests=settings.max_requests_per_minute, window_seconds=60) 
async def analyze_compatibility_simple(request: Request, data: JobAnalysisRequest):
    """Simplified compatibility analysis using Pydantic model"""
    try:
        if not analyzer:
            raise APIException(503, "Analyzer service unavailable")
        
        # Clean input data
        job_title = clean_text_safe(data.job_title)
        job_description = clean_text_safe(data.job_description)
        cv_text = clean_text_safe(data.cv_text)
        
        # Basic validation
        if len(job_title) < 5:
            raise APIException(422, "Job title must be at least 5 characters")
        
        if len(job_description.split()) < 10:
            raise APIException(422, "Job description must be at least 10 words")
        
        if len(cv_text.split()) < 100:
            raise APIException(422, "CV text must be at least 100 words")
        
        # Language validation
        from utils.language_utils import validate_english_text
        job_lang_check = validate_english_text(job_description, "Job Description")
        if not job_lang_check['is_english']:
            raise APIException(400, "Job description must be in English")
        
        # Generate analysis
        from utils.similarity_utils import generate_detailed_analysis_report
        results = generate_detailed_analysis_report(
            cv_text,
            job_title,
            job_description,
            analyzer
        )
        
        if not results:
            raise APIException(500, "Analysis computation failed")
        
        # Process results
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
        
        response_data = EnhancedCompatibilityResponse(
            overall_score=overall_score,
            text_similarity=results['text_similarity'],
            skill_match=results['skill_match'],
            experience_match=results['experience_match'],
            education_match=results['education_match'],
            industry_match=results['industry_match'],
            recommendation_level=recommendation_level,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            tips=tips,
            confidence_score=confidence_score
        )
        
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

@app.post("/analyze-compatibility")
@rate_limit(max_requests=settings.max_requests_per_minute, window_seconds=60)
async def analyze_compatibility(request: Request):
    """Analyze job compatibility with manual request parsing"""
    try:
        # Manual request body parsing to handle encoding issues
        try:
            body = await request.body()
            logger.info(f"Request body length: {len(body)}")
            
            if not body:
                logger.error("Empty request body")
                return JSONResponse(
                    status_code=422,
                    content={
                        "success": False,
                        "message": "Empty request body",
                        "errors": ["Request body is required"],
                        "timestamp": datetime.now().isoformat(),
                        "request_id": getattr(request.state, 'request_id', str(uuid.uuid4()))
                    }
                )
            
            # Try to decode body
            try:
                body_str = body.decode('utf-8', errors='replace')
                logger.info(f"Decoded body preview: {body_str[:200]}...")
            except Exception as decode_error:
                logger.error(f"Failed to decode body: {decode_error}")
                body_str = str(body, errors='replace')
            
            # Clean the JSON string but preserve JSON structure
            body_str = body_str.strip()
            
            # Parse JSON
            try:
                request_data = json.loads(body_str)
                logger.info(f"Successfully parsed JSON with keys: {list(request_data.keys())}")
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error: {json_error}")
                logger.error(f"Body content: {repr(body_str)}")
                return JSONResponse(
                    status_code=422,
                    content={
                        "success": False,
                        "message": "Invalid JSON format",
                        "errors": ["Request body must be valid JSON"],
                        "timestamp": datetime.now().isoformat(),
                        "request_id": getattr(request.state, 'request_id', str(uuid.uuid4()))
                    }
                )
            
        except Exception as e:
            logger.error(f"Failed to parse request body: {e}")
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "Invalid request format",
                    "errors": ["Could not parse request body"],
                    "timestamp": datetime.now().isoformat(),
                    "request_id": getattr(request.state, 'request_id', str(uuid.uuid4()))
                }
            )
        
        # Validate required fields
        required_fields = ['job_title', 'job_description', 'cv_text']
        for field in required_fields:
            if field not in request_data:
                return JSONResponse(
                    status_code=422,
                    content={
                        "success": False,
                        "message": f"Missing required field: {field}",
                        "errors": [f"Field '{field}' is required"],
                        "timestamp": datetime.now().isoformat(),
                        "request_id": getattr(request.state, 'request_id', str(uuid.uuid4()))
                    }
                )
        
        # Clean input data
        job_title = clean_text_safe(request_data['job_title'])
        job_description = clean_text_safe(request_data['job_description'])
        cv_text = clean_text_safe(request_data['cv_text'])
        
        # Basic validation
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
        
        if len(cv_text.split()) < 100:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "CV text too short",
                    "errors": ["CV text must be at least 100 words"],
                    "timestamp": datetime.now().isoformat(),
                    "request_id": getattr(request.state, 'request_id', str(uuid.uuid4()))
                }
            )
        
        if not analyzer:
            raise APIException(503, "Analyzer service unavailable")
        
        # Language validation
        from utils.language_utils import validate_english_text
        job_lang_check = validate_english_text(job_description, "Job Description")
        if not job_lang_check['is_english']:
            raise APIException(400, "Job description language validation failed")
        
        # Generate analysis
        from utils.similarity_utils import generate_detailed_analysis_report
        results = generate_detailed_analysis_report(
            cv_text,
            job_title,
            job_description,
            analyzer
        )
        
        if not results:
            raise APIException(500, "Analysis computation failed")
        
        # Process results
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
        
        response_data = EnhancedCompatibilityResponse(
            overall_score=overall_score,
            text_similarity=results['text_similarity'],
            skill_match=results['skill_match'],
            experience_match=results['experience_match'],
            education_match=results['education_match'],
            industry_match=results['industry_match'],
            recommendation_level=recommendation_level,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            tips=tips,
            confidence_score=confidence_score
        )
        
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