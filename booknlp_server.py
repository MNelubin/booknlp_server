"""
BookNLP GPU Service
Runs on host with GPU, accepts API calls for text processing
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from booknlp.booknlp import BookNLP
import os
import tempfile
import shutil
from typing import Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_SIZE = os.getenv("BOOKNLP_MODEL", "big")
MODELS_DIR = os.getenv("BOOKNLP_MODELS_DIR", "/models")
DATA_DIR = os.getenv("BOOKNLP_DATA_DIR", "/data")
API_PORT = int(os.getenv("API_PORT", "8888"))

# Global BookNLP instance
booknlp_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    global booknlp_model

    # Startup
    logger.info(f"Initializing BookNLP {MODEL_SIZE} model on GPU...")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")

    # Ensure directories exist
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    try:
        model_params = {
            "pipeline": "entity,quote,supersense,event,coref",
            "model": MODEL_SIZE
        }

        booknlp_model = BookNLP("en", model_params)
        logger.info("✓ BookNLP model loaded successfully!")
        logger.info(f"✓ Model: {MODEL_SIZE}")
        logger.info(f"✓ Pipeline: {model_params['pipeline']}")

    except Exception as e:
        logger.error(f"Failed to initialize BookNLP: {e}")
        raise

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down BookNLP GPU Service...")


app = FastAPI(
    title="BookNLP GPU Service",
    version="1.0.0",
    description="GPU-accelerated BookNLP microservice",
    lifespan=lifespan
)


class ExtractionRequest(BaseModel):
    text: str
    book_id: str
    pipeline: str = "entity,quote,supersense,event,coref"


class ExtractionResponse(BaseModel):
    status: str
    book_id: str
    message: str
    output_dir: str
    files: list[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    model: str
    gpu_enabled: bool
    cuda_available: Optional[bool] = None
    gpu_count: Optional[int] = None
    gpu_name: Optional[str] = None


@app.get("/", response_model=dict)
async def root():
    """Service information"""
    return {
        "service": "BookNLP GPU Service",
        "version": "1.0.0",
        "model": MODEL_SIZE,
        "status": "ready",
        "gpu": True,
        "endpoints": {
            "health": "/health",
            "extract": "/extract",
            "extract_file": "/extract_file",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with GPU status"""
    gpu_info = {
        "status": "healthy",
        "service": "BookNLP GPU Service",
        "model": MODEL_SIZE,
        "gpu_enabled": True
    }

    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        gpu_info["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        gpu_info["cuda_available"] = False
        logger.warning("PyTorch not available for GPU check")

    return gpu_info


@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest):
    """
    Extract entities from story text using GPU

    Args:
        request: ExtractionRequest with text, book_id, and optional pipeline

    Returns:
        ExtractionResponse with status and output directory
    """
    if not booknlp_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Processing book_id: {request.book_id}")
    logger.info(f"Text length: {len(request.text)} characters")

    # Create output directory for this job
    output_dir = Path(DATA_DIR) / f"booknlp_{request.book_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write text to temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(request.text)
        input_file = f.name

    try:
        # Run BookNLP
        logger.info(f"Running BookNLP {MODEL_SIZE} model...")
        booknlp_model.process(input_file, str(output_dir), request.book_id)

        # List generated files
        files = [f.name for f in output_dir.iterdir() if f.is_file()]

        logger.info(f"✓ Extraction completed. Generated {len(files)} files")

        return ExtractionResponse(
            status="success",
            book_id=request.book_id,
            message=f"Extraction completed for {len(request.text)} characters",
            output_dir=str(output_dir),
            files=files
        )

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        # Cleanup on error
        shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup input file
        if os.path.exists(input_file):
            os.unlink(input_file)


@app.post("/extract_file")
async def extract_file(file_path: str, book_id: str = "book"):
    """
    Extract from file path (file must be accessible in container)

    Args:
        file_path: Path to input text file
        book_id: Identifier for this book

    Returns:
        ExtractionResponse with status and output directory
    """
    if not booknlp_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    logger.info(f"Processing file: {file_path}")

    # Create output directory
    output_dir = Path(DATA_DIR) / f"booknlp_{book_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        booknlp_model.process(str(file_path_obj), str(output_dir), book_id)

        files = [f.name for f in output_dir.iterdir() if f.is_file()]

        return ExtractionResponse(
            status="success",
            book_id=book_id,
            message=f"Extracted from {file_path_obj.name}",
            output_dir=str(output_dir),
            files=files
        )

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{book_id}")
async def get_files(book_id: str):
    """
    List generated files for a specific book_id

    Args:
        book_id: Book identifier

    Returns:
        List of files in the output directory
    """
    output_dir = Path(DATA_DIR) / f"booknlp_{book_id}"

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"No results found for book_id: {book_id}")

    files = [
        {
            "name": f.name,
            "size": f.stat().st_size,
            "path": str(f)
        }
        for f in output_dir.iterdir()
        if f.is_file()
    ]

    return {"book_id": book_id, "files": files}


if __name__ == "__main__":
    import uvicorn

    # Use standard asyncio instead of uvloop to avoid permission errors in containers
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=API_PORT,
        log_level="info",
        loop="asyncio",  # Use standard asyncio, not uvloop
        access_log=True,
        use_colors=False
    )
