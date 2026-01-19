from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, JSONResponse
import cv2
import numpy as np

app = FastAPI()

def resize_to_width(img, target_w=1080):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)

def find_white_bands(gray, x1, x2, white_thr=235, min_run=10):
    """
    Біле поле = рядки, де середня яскравість >= white_thr протягом min_run рядків.
    Повертає список (start_y, end_y_exclusive).
    """
    row_mean = gray[:, x1:x2].mean(axis=1)
    is_white = row_mean >= white_thr

    bands = []
    h = gray.shape[0]
    y = 0
    while y < h:
        if not is_white[y]:
            y += 1
            continue
        start = y
        while y < h and is_white[y]:
            y += 1
        end = y
        if (end - start) >= min_run:
            bands.append((start, end))
    return bands

def edge_energy(gray, y1, y2, x1, x2):
    roi = gray[y1:y2, x1:x2]
    # Canny на ROI
    edges = cv2.Canny(roi, 60, 180)
    return float(edges.mean())  # 0..255, більше = більше "контенту"

def pick_best_gap(bands, h, y_min, y_max, min_gap_px, gray, x1, x2):
    """
    Перебираємо проміжки між сусідніми білими смугами і оцінюємо:
    - достатня висота
    - багато edges всередині (картинка/текст)
    - ближче донизу (частіше реклама там)
    """
    if len(bands) < 2:
        return None

    best = None  # (score, top, bottom)
    for i in range(len(bands) - 1):
        top = bands[i][1]
        bottom = bands[i + 1][0]

        # обмежити зоною пошуку
        if bottom <= y_min or top >= y_max:
            continue
        top = max(top, y_min)
        bottom = min(bottom, y_max)

        gap = bottom - top
        if gap < min_gap_px:
            continue

        # енергія контенту
        e = edge_energy(gray, top, bottom, x1, x2)

        # позиційний бонус (нижче — краще)
        center = (top + bottom) / 2.0
        pos_bonus = 1.0 + 0.3 * (center / float(h))  # 1.0..1.3

        score = (gap * 0.6 + e * 400.0) * pos_bonus

        if best is None or score > best[0]:
            best = (score, top, bottom)

    if best is None:
        return None
    return (best[1], best[2])

@app.get("/health")
def health():
    return {"ok": True, "opencv_version": cv2.__version__}

@app.post("/crop-ad")
async def crop_ad(
    file: UploadFile = File(...),
    debug: int = Query(0)
):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    img = resize_to_width(img, target_w=1080)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # центральна зона по ширині
    x1 = int(0.08 * w)
    x2 = int(0.92 * w)

    # зона по висоті (не чіпати верхній хедер, не лізти в navbar)
    y_min = int(0.15 * h)
    y_max = int(0.92 * h)

    # 1) знаходимо білі смуги
    bands = find_white_bands(gray, x1, x2, white_thr=235, min_run=10)

    # 2) вибираємо найкращий проміжок між двома смугами
    min_gap_px = int(0.18 * h)  # card має бути помітним по висоті
    gap = pick_best_gap(bands, h, y_min, y_max, min_gap_px, gray, x1, x2)

    if gap is None:
        if debug:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "no_ad_detected",
                    "h": h, "w": w,
                    "y_min": y_min, "y_max": y_max,
                    "bands_count": len(bands),
                    "bands_sample": bands[:8],
                    "params": {"white_thr": 235, "min_run": 10, "min_gap_px": min_gap_px},
                },
            )
        raise HTTPException(status_code=422, detail="no_ad_detected")

    top, bottom = gap

    # padding щоб “з запасом” включити підбанерний текст/CTA
    pad_top = int(0.01 * h)
    pad_bottom = int(0.02 * h)
    top = max(0, top - pad_top)
    bottom = min(y_max, bottom + pad_bottom)

    crop = img[top:bottom, 0:w]
    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=out.tobytes(), media_type="image/png")
