from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, JSONResponse
import cv2
import numpy as np

# Optional OCR dependency:
# pip install pytesseract
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    pytesseract = None
    OCR_AVAILABLE = False

app = FastAPI()

# ---------- Core image utils ----------

def resize_to_width(img, target_w=1080):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def detect_theme(gray, w, h):
    x1 = int(0.20 * w)
    x2 = int(0.80 * w)
    y1 = int(0.12 * h)
    y2 = int(0.28 * h)
    roi = gray[y1:y2, x1:x2]
    med = float(np.median(roi)) if roi.size else float(np.median(gray))
    return "light" if med >= 150.0 else "dark"


def find_bg_bands(
    gray, x1, x2, theme,
    min_run=24,
    white_thr=235, white_frac=0.985,
    black_thr=30, black_frac=0.985
):
    roi = gray[:, x1:x2]

    if theme == "light":
        frac = (roi >= white_thr).mean(axis=1)
        is_bg = frac >= white_frac
    else:
        frac = (roi <= black_thr).mean(axis=1)
        is_bg = frac >= black_frac

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


# ---------- OCR / keyword logic ----------

SPONSORED_KEYWORDS = [
    # English
    "sponsored", "ad", "advertisement",
    # Ukrainian / Russian
    "реклама", "спонсоровано", "спонсороване", "спонсорированный", "спонсорировано",
    # Polish / Czech / Slovak
    "reklama", "sponsorowane", "sponzorované", "sponzorovaný",
    # German / French / Spanish / Italian / Portuguese / Romanian / Turkish
    "anzeige", "gesponsert",
    "sponsorisé", "publicité",
    "patrocinado", "anuncio",
    "sponsorizzato",
    "patrocinado", "anúncio",
    "sponsorizat", "publicitate",
    "sponsorlu", "reklam",
]

def normalize_text(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip())


def ocr_find_sponsored_bbox(gray_roi, tesseract_langs: str):
    """
    Returns (found: bool, max_y: int, debug_words: list)
    max_y = максимальна нижня координата слова-ключа у ROI (в пікселях ROI)
    """
    if not OCR_AVAILABLE:
        return False, None, []

    # Mild preprocessing for OCR
    img = gray_roi.copy()
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    config = "--oem 1 --psm 6"
    try:
        data = pytesseract.image_to_data(img, lang=tesseract_langs, config=config, output_type=pytesseract.Output.DICT)
    except Exception:
        # language packs not installed, fallback to eng
        data = pytesseract.image_to_data(img, lang="eng", config=config, output_type=pytesseract.Output.DICT)

    found = False
    max_y = None
    debug_words = []

    n = len(data.get("text", []))
    for i in range(n):
        txt = normalize_text(data["text"][i] or "")
        if not txt:
            continue

        # store for debugging (bounded)
        if len(debug_words) < 50:
            debug_words.append(txt)

        # direct match or substring match (handles OCR noise like "sponsor ed")
        for kw in SPONSORED_KEYWORDS:
            kw_n = normalize_text(kw)
            if kw_n and (kw_n in txt or txt in kw_n):
                found = True
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                y_bottom = y + h
                if max_y is None or y_bottom > max_y:
                    max_y = y_bottom
                break

    return found, max_y, debug_words


# ---------- API ----------

@app.get("/health")
def health():
    return {"ok": True, "opencv_version": cv2.__version__, "ocr_available": OCR_AVAILABLE}


@app.post("/crop-ad")
async def crop_ad(
    file: UploadFile = File(...),
    debug: bool = Query(False),
    # if strict=true -> behave like before (422 no_ad_detected)
    strict: bool = Query(False),
    # how far below the detected card we allow OCR to inspect (fraction of screen height)
    ocr_scan_h: float = Query(0.22),
    # how much extra we extend after the keyword baseline (fraction of screen height)
    ocr_extend_h: float = Query(0.10),
    # tesseract languages (must be installed in the container)
    tesseract_langs: str = Query("eng+ukr+rus+deu+fra+spa+ita+por+pol+ron+tur"),
):
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

    # 2) Theme
    theme = detect_theme(gray, w, h)

    # 3) Central width
    x1 = int(0.15 * w)
    x2 = int(0.85 * w)

    # 4) Detect bands and gap
    bands = find_bg_bands(
        gray, x1, x2, theme,
        min_run=24,
        white_thr=235, white_frac=0.985,
        black_thr=30, black_frac=0.985
    )

    gap = pick_best_gap_between_bands(
        bands,
        h,
        min_gap_px=int(0.18 * h),
        prefer_lower_half=True
    )

    if gap is None:
        if strict:
            raise HTTPException(status_code=422, detail="no_ad_detected")
        # fallback: return original (so n8n doesn't fail)
        ok, out0 = cv2.imencode(".png", img)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        return Response(content=out0.tobytes(), media_type="image/png", headers={"X-Crop-Status": "no_ad_detected"})

    top, bottom = gap

    # 5) Padding
    pad_top = int(0.01 * h)
    pad_bottom = int(0.01 * h)
    top = max(0, top - pad_top)
    bottom = min(h, bottom + pad_bottom)

    # 6) Protect navbar
    bottom_limit = int(0.92 * h)
    bottom = min(bottom, bottom_limit)

    # 7) OCR below the creative to decide if we need to include sponsored text
    ocr_used = False
    sponsored_found = False
    sponsored_y = None
    debug_words = []

    if OCR_AVAILABLE and bottom < bottom_limit:
        scan_top = bottom
        scan_bottom = min(bottom_limit, bottom + int(ocr_scan_h * h))
        if scan_bottom > scan_top + 10:
            roi = gray[scan_top:scan_bottom, x1:x2]
            sponsored_found, sponsored_y, debug_words = ocr_find_sponsored_bbox(roi, tesseract_langs)
            ocr_used = True

            if sponsored_found and sponsored_y is not None:
                # extend to include the line and some extra UI below it
                new_bottom = scan_top + sponsored_y + int(ocr_extend_h * h)
                bottom = min(bottom_limit, new_bottom)

    crop = img[top:bottom, 0:w]
    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    if debug:
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "theme": theme,
                "ocr_available": OCR_AVAILABLE,
                "ocr_used": ocr_used,
                "sponsored_found": sponsored_found,
                "top": int(top),
                "bottom": int(bottom),
                "h": int(h),
                "w": int(w),
                "tesseract_langs": tesseract_langs,
                "debug_words_sample": debug_words[:30],
            },
        )

    return Response(
        content=out.tobytes(),
        media_type="image/png",
        headers={
            "X-Crop-Status": "cropped",
            "X-Theme": theme,
            "X-OCR": "1" if ocr_used else "0",
            "X-Sponsored": "1" if sponsored_found else "0",
        },
    )
