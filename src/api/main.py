from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .endpoints import router, load_models
import contextlib

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models before app starts
    load_models()
    yield
    # Clean up if necessary

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time Recommendation / Fraud Detection Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Allow CORS for the dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running. Go to /docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
