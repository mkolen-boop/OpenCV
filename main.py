from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import cv2
import numpy as np

app = FastAPI()


# -----------------------
# Utilities
# -----------------------
def resize_to_width(img, target_w=1080):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def detect_theme(gray, w, h):
    """
    Robust-enough theme detection: median brightness in upper-central area.
    """
    x1 = int(0.20 * w)
    x2 = int(0.80 * w)
    y1 = int(0.12 * h)
    y2 = int(0.28 * h)
    roi = gray[y1:y2, x1:x2]
    med = float(np.median(roi)) if roi.size else float(np.median(gray))
    return "light" if med >= 150.0 else "dark"


def merge_close_bands(bands, max_gap_px=10):
    """
    Merge background bands if they are separated by a very small gap.
    Helps when a band is fragmented by tiny UI artifacts.
    """
    if not bands:
        return []
    merged = [bands[0]]
    for s, e in bands[1:]:
        ps, pe = merged[-1]
        if s - pe <= max_gap_px:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def find_bg_bands(gray, x1, x2, theme,
                  min_run=18,
                  # light
                  white_thr=235, white_frac=0.990,
                  # dark
                  black_thr=35, black_frac=0.990,
                  # edge gating (prevents "text on white" from being treated as pure background)
                  edge_thr=18.0, edge_low_frac=0.90):
    """
    Background band = rows that are:
      - mostly background color (light: very bright; dark: very dark)
      - AND have low edge energy (to exclude text/buttons on same background)

    Returns list of (start_y, end_y exclusive).
    """
    roi = gray[:, x1:x2].astype(np.uint8)

    # Background-by-color
    if theme == "light":
        frac_bg = (roi >= white_thr).mean(axis=1)
        is_bg_color = frac_bg >= white_frac
    else:
        frac_bg = (roi <= black_thr).mean(axis=1)
        is_bg_color = frac_bg >= black_frac

    # Background-by-edges
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)  # float32

    low_edge_frac = (mag <= edge_thr).mean(axis=1)
    is_bg_edge = low_edge_frac >= edge_low_frac

    is_bg = is_bg_color & is_bg_edge

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


def pick_gap_skip_inner_stripes(bands, h,
                                min_card_px,
                                min_gap_px,
                                min_bottom_band_px,
                                prefer_lower_half=True):
    """
    Core fix for YouTube:
    - internal white padding stripe exists between creative and text -> ignore it
    - choose a gap only if it's tall enough AND the next background band is "real" (thick enough)
    - additionally enforce that bottom is at least min_card_px below top

    Returns (top, bottom) or None.
    """
    if len(bands) < 2:
        return None

    best = None  # (score, top, bottom)

    for i in range(len(bands) - 1):
        top = bands[i][1]
        bottom = bands[i + 1][0]

        gap = bottom - top
        bottom_band_h = bands[i + 1][1] - bands[i + 1][0]

        # 1) Ignore tiny inner separators by requiring "card height" from this top
        if bottom < top + min_card_px:
            continue

        # 2) Require gap itself to be meaningful
        if gap < min_gap_px:
            continue

        # 3) Require the LOWER band to look like a real inter-card background band
        if bottom_band_h < min_bottom_band_px:
            continue

        # Scoring: prefer larger gaps, and prefer lower-half placement
        center = (top + bottom) / 2.0
        score = float(gap)
        if prefer_lower_half:
            score *= 1.25 if center >= 0.45 * h else 0.85

        if best is None or score > best[0]:
            best = (score, top, bottom)

    if best is None:
        return None
    return (best[1], best[2])


# -----------------------
# API
# -----------------------
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

    # 1) Normalize size
    img = resize_to_width(img, target_w=1080)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Theme
    theme = detect_theme(gray, w, h)

    # 3) Work on central width (avoid avatar/menu columns)
    x1 = int(0.18 * w)
    x2 = int(0.82 * w)

    # 4) Find background bands (robust vs text)
    bands = find_bg_bands(
        gray, x1, x2, theme,
        min_run=18,
        white_thr=235, white_frac=0.990,
        black_thr=35, black_frac=0.990,
        edge_thr=18.0, edge_low_frac=0.90
    )

    # Merge fragmented bands (helps stability)
    bands = merge_close_bands(bands, max_gap_px=10)

    # 5) Pick the correct gap (skip inner padding stripe)
    gap = pick_gap_skip_inner_stripes(
        bands, h,
        min_card_px=int(0.25 * h),        # MUST be tall enough to include creative + text
        min_gap_px=int(0.18 * h),         # ignore tiny gaps
        min_bottom_band_px=int(0.02 * h), # lower band must be "real" background (not a thin inner stripe)
        prefer_lower_half=True
    )

    if gap is None:
        raise HTTPException(status_code=422, detail="no_ad_detected")

    top, bottom = gap

    # 6) Padding and navbar protection
    pad_top = int(0.01 * h)
    pad_bottom = int(0.015 * h)
    top = max(0, top - pad_top)
    bottom = min(h, bottom + pad_bottom)

    bottom = min(bottom, int(0.92 * h))  # avoid bottom navigation

    crop = img[top:bottom, 0:w]

    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=out.tobytes(), media_type="image/png")
