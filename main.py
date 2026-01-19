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


def row_mean(gray, x1, x2):
    return gray[:, x1:x2].mean(axis=1)


def row_var(gray, x1, x2):
    roi = gray[:, x1:x2].astype(np.float32)
    return roi.var(axis=1)


def find_white_bands(gray, x1, x2, white_thr=245, min_run=18):
    m = row_mean(gray, x1, x2)
    is_white = m >= white_thr

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
            bands.append((start, end))  # end exclusive
    return bands


def nearest_band_above(bands, y):
    # band with end <= y, choose the closest
    cand = [b for b in bands if b[1] <= y]
    if not cand:
        return None
    return max(cand, key=lambda b: b[1])


def nearest_band_below(bands, y):
    # band with start >= y, choose the closest
    cand = [b for b in bands if b[0] >= y]
    if not cand:
        return None
    return min(cand, key=lambda b: b[0])


def find_media_block(gray, x1, x2, y_min, y_max):
    """
    Шукаємо вікно по Y з максимальною "текстурністю" (variance),
    що зазвичай відповідає великій картинці/медіа в ad card.
    """
    h = gray.shape[0]
    y_min = max(0, min(h - 1, y_min))
    y_max = max(0, min(h, y_max))
    if y_max - y_min < 200:
        return None

    v = row_var(gray, x1, x2)
    v = v[y_min:y_max]

    # Згладжування, щоб не реагувати на дрібний шум
    v_s = cv2.GaussianBlur(v.reshape(-1, 1), (1, 31), 0).reshape(-1)

    # Порог по відносному рівню (беремо верхню "енергію")
    thr = float(np.percentile(v_s, 80))  # можна 75-85
    is_media = v_s >= thr

    # Найдовший contiguous run як медіа-блок
    best = None  # (len, start, end)
    i = 0
    n = len(is_media)
    while i < n:
        if not is_media[i]:
            i += 1
            continue
        s = i
        while i < n and is_media[i]:
            i += 1
        e = i
        run_len = e - s
        if best is None or run_len > best[0]:
            best = (run_len, s, e)

    if best is None:
        return None

    run_len, s, e = best

    # Мінімальна висота медіа (щоб не зловити текст)
    if run_len < int(0.12 * (y_max - y_min)):
        return None

    # Повертаємо в абсолютних координатах
    return (y_min + s, y_min + e)


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

    img = resize_to_width(img, target_w=1080)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Працюємо по центральній ширині
    x1 = int(0.08 * w)
    x2 = int(0.92 * w)

    # Зона пошуку реклами: нижні ~65% екрану (щоб не чіпати верхній хедер YouTube)
    y_search_min = int(0.20 * h)
    y_search_max = int(0.92 * h)

    bands = find_white_bands(gray, x1, x2, white_thr=245, min_run=18)

    media = find_media_block(gray, x1, x2, y_search_min, y_search_max)
    if media is None:
        raise HTTPException(status_code=422, detail="no_ad_detected")

    media_top, media_bottom = media

    top_band = nearest_band_above(bands, media_top)
    bot_band = nearest_band_below(bands, media_bottom)

    # Якщо зовнішні білі смуги не знайдені — fallback рамки від медіа
    if top_band is None:
        top = max(0, media_top - int(0.08 * h))
    else:
        top = top_band[1]

    if bot_band is None:
        bottom = min(int(0.92 * h), media_bottom + int(0.25 * h))  # включає текст/CTA
    else:
        bottom = bot_band[0]

    # Паддінг, щоб “запасом” включити весь card
    top = max(0, top - int(0.01 * h))
    bottom = min(int(0.92 * h), bottom + int(0.01 * h))

    # sanity
    if bottom - top < int(0.12 * h):
        raise HTTPException(status_code=422, detail="no_ad_detected")

    crop = img[top:bottom, 0:w]

    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=out.tobytes(), media_type="image/png")
