from fastapi import FastAPI
import cv2

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "opencv_version": cv2.__version__}
