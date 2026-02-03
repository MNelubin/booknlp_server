"""
BookNLP GPU Service
Runs on host with GPU, accepts API calls for text processing
"""
# Monkey patch torch.load to remove position_ids from state_dict
# This fixes compatibility with transformers 4.x+
import torch
_original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    result = _original_torch_load(f, *args, **kwargs)
    # Remove position_ids if present (for transformers 4.x+ compatibility)
    if isinstance(result, dict) and "bert.embeddings.position_ids" in result:
        del result["bert.embeddings.position_ids"]
    return result

torch.load = patched_torch_load

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
import asyncio
from threading import Lock

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
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "300"))  # 5 minutes default
MODEL_WARMUP = os.getenv("MODEL_WARMUP", "true").lower() == "true"  # Warmup on startup

# Global BookNLP instance and state
booknlp_model: Optional[BookNLP] = None
_model_lock = Lock()
_last_activity_time: Optional[float] = None
_unload_task: Optional[asyncio.Task] = None
_model_loaded = False


def _load_model() -> None:
    """Load BookNLP model (thread-safe)"""
    global booknlp_model, _model_loaded

    with _model_lock:
        if _model_loaded:
            logger.debug("Model already loaded, skipping")
            return

        logger.info("=" * 60)
        logger.info(f"Loading BookNLP {MODEL_SIZE} model on GPU...")
        logger.info("=" * 60)
        logger.info(f"Models directory: {MODELS_DIR}")
        logger.info(f"Model size: {MODEL_SIZE}")

        # Check CUDA availability
        try:
            import torch
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        except Exception as e:
            logger.warning(f"Could not check CUDA status: {e}")

        try:
            model_params = {
                "pipeline": "entity,quote,supersense,event,coref",
                "model": MODEL_SIZE
            }

            logger.info(f"Initializing BookNLP with params: {model_params}")
            logger.info("This may take a while on first run (downloading models)...")

            import time
            start_time = time.time()

            booknlp_model = BookNLP("en", model_params)

            load_time = time.time() - start_time
            _model_loaded = True

            logger.info("=" * 60)
            logger.info("✓ BookNLP model loaded successfully!")
            logger.info(f"✓ Model: {MODEL_SIZE}")
            logger.info(f"✓ Pipeline: {model_params['pipeline']}")
            logger.info(f"✓ Load time: {load_time:.2f} seconds")

            # Report memory usage after loading
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"✓ GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                    logger.info(f"✓ GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            except Exception as e:
                logger.warning(f"Could not check GPU memory: {e}")

            logger.info("=" * 60)

        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"✗ Failed to load BookNLP model!")
            logger.error(f"✗ Error: {e}")
            logger.error(f"✗ Error type: {type(e).__name__}")
            import traceback
            logger.error(f"✗ Traceback:\n{traceback.format_exc()}")
            logger.error("=" * 60)
            raise


def _unload_model() -> None:
    """Unload BookNLP model to free memory (thread-safe)"""
    global booknlp_model, _model_loaded, _last_activity_time, _unload_task

    with _model_lock:
        if not _model_loaded:
            return

        logger.info("Unloading BookNLP model to free memory...")

        try:
            # Delete the model instance
            del booknlp_model
            booknlp_model = None
            _model_loaded = False
            _last_activity_time = None
            _unload_task = None

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("✓ CUDA cache cleared")
            except ImportError:
                pass

            logger.info("✓ Model unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading model: {e}")


async def _schedule_model_unload() -> None:
    """Schedule model unload after timeout period"""
    global _unload_task

    # Cancel existing task if any
    if _unload_task and not _unload_task.done():
        _unload_task.cancel()

    # Create new unload task
    _unload_task = asyncio.create_task(_model_unload_worker())


async def _model_unload_worker() -> None:
    """Worker task that unloads model after idle timeout"""
    global _last_activity_time

    try:
        logger.info(f"Model unload scheduled in {MODEL_IDLE_TIMEOUT} seconds...")
        await asyncio.sleep(MODEL_IDLE_TIMEOUT)

        # Check if there was any activity during sleep
        import time
        current_time = time.time()
        if _last_activity_time and (current_time - _last_activity_time) >= MODEL_IDLE_TIMEOUT:
            logger.info(f"No activity for {MODEL_IDLE_TIMEOUT} seconds, unloading model...")
            _unload_model()
        else:
            logger.debug("Activity detected, cancelling model unload")

    except asyncio.CancelledError:
        logger.debug("Model unload task cancelled")
        raise


def _update_activity() -> None:
    """Update last activity timestamp and schedule model unload"""
    global _last_activity_time

    import time
    _last_activity_time = time.time()

    # Schedule unload task in background
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_schedule_model_unload())
    except RuntimeError:
        # No event loop running (shouldn't happen in FastAPI)
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info("")
    logger.info("=" * 60)
    logger.info("Starting BookNLP GPU Service...")
    logger.info("=" * 60)
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Model idle timeout: {MODEL_IDLE_TIMEOUT} seconds")
    logger.info(f"Model warmup on startup: {MODEL_WARMUP}")

    # Ensure directories exist
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Warmup: load model once to cache models, then unload
    if MODEL_WARMUP:
        logger.info("")
        logger.info("🔥 Warming up: loading model to cache...")
        logger.info("This will download models on first run and cache them.")
        logger.info("After caching, model will be unloaded to free GPU memory.")
        logger.info("")

        try:
            _load_model()
            logger.info("✓ Model warmup completed successfully!")

            # Immediately unload to free memory
            logger.info("")
            logger.info("Unloading model after warmup to free GPU memory...")
            _unload_model()
            logger.info("✓ Model unloaded. Ready for lazy loading on first request.")
            logger.info("")

        except Exception as e:
            logger.error(f"✗ Model warmup failed: {e}")
            logger.error("Service will start anyway, model will load on first request.")
    else:
        logger.info("Model warmup disabled. Model will load on first request.")

    logger.info("=" * 60)
    logger.info("✓ BookNLP GPU Service is ready!")
    logger.info("=" * 60)
    logger.info("")

    yield

    # Shutdown
    logger.info("")
    logger.info("Shutting down BookNLP GPU Service...")
    _unload_model()
    logger.info("Shutdown complete.")


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
    model_loaded: bool = False
    cuda_available: Optional[bool] = None
    gpu_count: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_cached_mb: Optional[float] = None


@app.get("/", response_model=dict)
async def root():
    """Service information"""
    return {
        "service": "BookNLP GPU Service",
        "version": "1.1.0",
        "model": MODEL_SIZE,
        "status": "ready",
        "gpu": True,
        "lazy_loading": True,
        "idle_timeout_seconds": MODEL_IDLE_TIMEOUT,
        "model_loaded": _model_loaded,
        "endpoints": {
            "health": "/health",
            "extract": "/extract",
            "extract_file": "/extract_file",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with GPU status and model state"""
    gpu_info = {
        "status": "healthy",
        "service": "BookNLP GPU Service",
        "model": MODEL_SIZE,
        "gpu_enabled": True,
        "model_loaded": _model_loaded
    }

    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        gpu_info["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            if _model_loaded:
                # Get memory usage if model is loaded
                gpu_info["gpu_memory_used_mb"] = torch.cuda.memory_allocated(0) / 1024 / 1024
                gpu_info["gpu_memory_cached_mb"] = torch.cuda.memory_reserved(0) / 1024 / 1024
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
    # Load model if not loaded (lazy loading)
    if not _model_loaded:
        logger.info("Model not loaded, loading now...")
        _load_model()
    elif not booknlp_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Update activity and schedule unload
    _update_activity()

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
    # Load model if not loaded (lazy loading)
    if not _model_loaded:
        logger.info("Model not loaded, loading now...")
        _load_model()
    elif not booknlp_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Update activity and schedule unload
    _update_activity()

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


@app.post("/model/load")
async def load_model():
    """
    Manually load the BookNLP model

    Returns:
        Status message
    """
    if _model_loaded:
        return {"status": "already_loaded", "message": "Model is already loaded"}

    try:
        _load_model()
        return {"status": "loaded", "message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/unload")
async def unload_model():
    """
    Manually unload the BookNLP model to free GPU memory

    Returns:
        Status message
    """
    if not _model_loaded:
        return {"status": "not_loaded", "message": "Model is not loaded"}

    try:
        _unload_model()
        return {"status": "unloaded", "message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
