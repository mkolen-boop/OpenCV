from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
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


def detect_theme(gray, w, h):
    """
    Оцінюємо яскравість UI-фону у верхній центральній зоні (без статусбару).
    """
    x1 = int(0.20 * w)
    x2 = int(0.80 * w)
    y1 = int(0.12 * h)
    y2 = int(0.28 * h)
    roi = gray[y1:y2, x1:x2]
    med = float(np.median(roi)) if roi.size else float(np.median(gray))
    return "light" if med >= 150.0 else "dark"


def find_bg_bands_youtube(
    gray, x1, x2, theme,
    min_run=20,
    # light thresholds
    white_thr=235,
    white_frac=0.990,
    # dark thresholds
    black_thr=35,
    black_frac=0.990,
    # edge thresholding (critical)
    edge_thr=18.0,
    edge_low_frac=0.90
):
    """
    Фонова смуга = рядки, де:
      1) більшість пікселів близькі до фону (дуже світлі у light / дуже темні у dark)
      2) і одночасно мало ребер (edge energy низька) -> текст/іконки відсікаються

    Повертає (start_y, end_y exclusive) для кожної фонової смуги.
    """
    roi = gray[:, x1:x2].astype(np.uint8)

    # 1) "background-like" mask per row
    if theme == "light":
        frac_bg = (roi >= white_thr).mean(axis=1)
        is_bg_by_color = frac_bg >= white_frac
    else:
        frac_bg = (roi <= black_thr).mean(axis=1)
        is_bg_by_color = frac_bg >= black_frac

    # 2) edge energy per row (Sobel magnitude)
    # Use CV_32F for stable magnitudes
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)  # float32

    # Row is "low-edge" if most pixels have magnitude below edge_thr
    low_edge_frac = (mag <= edge_thr).mean(axis=1)
    is_low_edge = low_edge_frac >= edge_low_frac

    is_bg = is_bg_by_color & is_low_edge

    bands = []
    h = gray.shape[0]
    y = 0
    while y < h:
        if not is_bg[y]:
            y += 1
            continue
        start = y
        while y < h and is_bg[y]:
            y += 1
        end = y
        if (end - start) >= min_run:
            bands.append((start, end))
    return bands


def pick_best_gap_between_bands(bands, h, min_gap_px, prefer_lower_half=True):
    """
    Обираємо найкращий gap між сусідніми фоновими смугами.
    """
    if len(bands) < 2:
        return None

    best = None  # (score, top, bottom)
    for i in range(len(bands) - 1):
        top = bands[i][1]
        bottom = bands[i + 1][0]
        gap = bottom - top
        if gap < min_gap_px:
            continue

        center = (top + bottom) / 2.0
        score = float(gap)
        if prefer_lower_half:
            score *= 1.25 if center >= 0.45 * h else 0.85

        if best is None or score > best[0]:
            best = (score, top, bottom)

    if best is None:
        return None
    return (best[1], best[2])


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

    # Normalize for different phones
    img = resize_to_width(img, target_w=1080)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    theme = detect_theme(gray, w, h)

    # Central width to avoid avatar/menu columns
    x1 = int(0.18 * w)
    x2 = int(0.82 * w)

    # Find background bands (now robust vs text)
    bands = find_bg_bands_youtube(
        gray, x1, x2, theme,
        min_run=20,
        white_thr=235, white_frac=0.990,
        black_thr=35, black_frac=0.990,
        edge_thr=18.0, edge_low_frac=0.90
    )

    # Pick best "card gap" (image + text)
    gap = pick_best_gap_between_bands(
        bands,
        h,
        min_gap_px=int(0.20 * h),
        prefer_lower_half=True
    )

    if gap is None:
        raise HTTPException(status_code=422, detail="no_ad_detected")

    top, bottom = gap

    # Padding
    pad_top = int(0.01 * h)
    pad_bottom = int(0.015 * h)
    top = max(0, top - pad_top)
    bottom = min(h, bottom + pad_bottom)

    # Avoid bottom navigation bar
    bottom = min(bottom, int(0.92 * h))

    crop = img[top:bottom, 0:w]

    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=out.tobytes(), media_type="image/png")
