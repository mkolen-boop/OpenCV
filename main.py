from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import cv2
import numpy as np

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "opencv_version": cv2.__version__}

@app.post("/crop-ad")
async def crop_ad(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    h, w = img.shape[:2]

    # TEMPORARY: просто повертаємо нижню частину (sanity-check)
    crop = img[int(0.3 * h):int(0.9 * h), 0:w]

    _, out = cv2.imencode(".png", crop)
    return Response(content=out.tobytes(), media_type="image/png")

