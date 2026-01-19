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
            score *= 1.2 if center >= 0.50 * h else 0.8

        if best is None or score > best[0]:
            best = (score, top, bottom)

    if best is None:
        return None
    return (best[1], best[2])


def first_white_band_at_or_below(bands, y):
    for b in bands:
        if b[0] >= y:
            return b
    return None


def is_text_block(gray, y1, y2, x1, x2):
    """
    Евристика "це текстовий блок під рекламою":
    - фон світлий (mean високе)
    - картинка не "шумна" (std невисокий)
    - є помірна кількість контурів букв (edges не нульові, але не як у фото)
    """
    if y2 <= y1:
        return False

    roi = gray[y1:y2, x1:x2]
    mean = float(roi.mean())
    std = float(roi.std())

    edges = cv2.Canny(roi, 60, 180)
    edge_density = float((edges > 0).mean())  # 0..1

    # Текстовий блок зазвичай:
    # - дуже світлий фон
    # - std відносно невеликий
    # - edge_density маленький, але не 0 (букви)
    if mean >= 215 and std <= 45 and 0.002 <= edge_density <= 0.06:
        return True

    return False


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

    # 1) Normalize
    img = resize_to_width(img, target_w=1080)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) центральна ширина
    x1 = int(0.08 * w)
    x2 = int(0.92 * w)

    # 3) базовий gap між білими полями
    bands = find_white_bands(gray, x1, x2, white_thr=245, min_run=18)
    gap = pick_best_gap_between_bands(
        bands,
        h,
        min_gap_px=int(0.15 * h),
        prefer_lower_half=True
    )
    if gap is None:
        raise HTTPException(status_code=422, detail="no_ad_detected")

    top, bottom = gap

    # 4) Спроба додати підбанерний текст (АЛЕ лише якщо це текст)
    nav_limit = int(0.92 * h)

    # Шукаємо білу смугу одразу після bottom
    b1 = first_white_band_at_or_below(bands, bottom)
    if b1 is not None:
        # Кандидат-текстовий блок починається після цієї білої смуги
        text_start = b1[1]

        # Шукаємо наступну білу смугу нижче — це може бути кінець "текстового" блоку
        b2 = first_white_band_at_or_below(bands, text_start + 1)

        # Якщо знайшли b2 — кандидат це [text_start, b2.start]
        # Якщо не знайшли — беремо обмежений шматок (щоб не залетіти в інший контент)
        if b2 is not None:
            text_end = min(b2[0], nav_limit)
        else:
            text_end = min(text_start + int(0.18 * h), nav_limit)

        # ДОДАЄМО ВНИЗ ТІЛЬКИ ЯКЩО ЦЕ СХОЖЕ НА ТЕКСТ,
        # і висота блоку не надто велика (щоб не захопити наступне прев’ю)
        if (text_end - text_start) >= int(0.03 * h) and (text_end - text_start) <= int(0.20 * h):
            if is_text_block(gray, text_start, text_end, x1, x2):
                bottom = text_end  # розширили
            # else: не чіпаємо bottom (значить там вже інше фото/контент)

    # 5) Padding + clamp
    pad_top = int(0.01 * h)
    pad_bottom = int(0.01 * h)
    top = max(0, top - pad_top)
    bottom = min(nav_limit, bottom + pad_bottom)

    crop = img[top:bottom, 0:w]

    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=out.tobytes(), media_type="image/png")
