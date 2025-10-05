from fastapi import FastAPI
from Routes import router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os

load_dotenv()


app=FastAPI()
app.include_router(router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_credentials=True)
app.mount("/assets",StaticFiles(directory="dist/assets"),name="assets")


@app.get("/{full_path:path}")
async def main():
    return FileResponse("dist/index.html")
