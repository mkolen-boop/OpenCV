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


def find_white_bands(gray, x1, x2, white_thr=245, min_run=18):
    """
    Повертає список білих горизонтальних смуг (start_y, end_y exclusive).
    Білою вважаємо смугу, де середня яскравість рядка >= white_thr
    протягом min_run рядків.
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


def pick_best_gap_between_bands(bands, h, min_gap_px, prefer_lower_half=True):
    """
    Беремо проміжки між сусідніми білими смугами та обираємо найкращий:
    - gap >= min_gap_px
    - бажано в нижній половині екрану
    """
    if len(bands) < 2:
        return None

    best = None  # (score, top, bottom)
    for i in range(len(bands) - 1):
        top = bands[i][1]      # кінець верхньої білої смуги
        bottom = bands[i+1][0] # початок нижньої білої смуги
        gap = bottom - top
        if gap < min_gap_px:
            continue

        center = (top + bottom) / 2.0
        score = float(gap)

        if prefer_lower_half:
            score *= 1.2 if center >= 0.50 * h else 0.8

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

    # 1) Normalize (різні телефони)
    img = resize_to_width(img, target_w=1080)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Працюємо по центральній ширині, щоб зменшити вплив країв/іконок
    x1 = int(0.08 * w)
    x2 = int(0.92 * w)

    # 3) Знаходимо білі поля та беремо проміжок між двома сусідніми білими полями
    bands = find_white_bands(gray, x1, x2, white_thr=245, min_run=18)
    gap = pick_best_gap_between_bands(
        bands,
        h,
        min_gap_px=int(0.15 * h),  # мінімальна висота "card"
        prefer_lower_half=True
    )

    if gap is None:
        raise HTTPException(status_code=422, detail="no_ad_detected")

    top, bottom = gap

    # 4) Padding, щоб взяти весь банер + те, що під ним належить
    pad_top = int(0.01 * h)
    pad_bottom = int(0.01 * h)
    top = max(0, top - pad_top)
    bottom = min(h, bottom + pad_bottom)

    # 5) Не залазимо в нижній navbar
    bottom = min(bottom, int(0.92 * h))

    crop = img[top:bottom, 0:w]

    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=out.tobytes(), media_type="image/png")
