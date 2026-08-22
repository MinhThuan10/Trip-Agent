from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.src.api.v1.endpoints import chat, admin
from app.src.config.settings import settings
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files & templates if needed
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["Chat"])
app.include_router(admin.router, prefix=f"{settings.API_V1_STR}/admin", tags=["Admin"])

templates = Jinja2Templates(directory="app/templates")
@app.get("/")
def root():
    return {"message": "Welcome to Trip Agent System API", "docs": "/docs", "admin dashboard": "/admin"}

templates = Jinja2Templates(directory="app/templates")
@app.get("/admin", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="admin.html"
    )
