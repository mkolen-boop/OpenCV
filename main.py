from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import cv2
import numpy as np

app = FastAPI()


# -----------------------
# Basic utilities
# -----------------------
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


def find_bg_bands(gray, x1, x2, theme,
                  min_run=18,
                  white_thr=235, white_frac=0.985,
                  black_thr=30, black_frac=0.985):
    roi = gray[:, x1:x2]

    if theme == "light":
        frac = (roi >= white_thr).mean(axis=1)
        is_bg = frac >= white_frac
    else:
        frac = (roi <= black_thr).mean(axis=1)
        is_bg = frac >= black_frac

    bands = []
    H = gray.shape[0]
    y = 0
    while y < H:
        if not is_bg[y]:
            y += 1
            continue
        start = y
        while y < H and is_bg[y]:
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


# -----------------------
# Edge-based "text block" extension (no OCR)
# -----------------------
def row_edge_strength(gray, x1, x2):
    """
    Returns per-row edge strength (mean Sobel magnitude in ROI).
    """
    roi = gray[:, x1:x2].astype(np.uint8)
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag.mean(axis=1)  # shape: (H,)


def extend_bottom_to_include_text(gray, x1, x2, top, bottom_guess, bottom_limit):
    """
    We assume:
      - top is correct start of card (just below a background band).
      - bottom_guess often lands at the first background padding stripe after image.
    We:
      1) find image_end as first place where background-like rows resume (near bottom_guess)
      2) find text_start: first strong-edge region after image_end (text on background)
      3) find text_end: where edges drop and stay low -> end of card content
    If we can't find a consistent text block, return bottom_guess.
    """
    H = gray.shape[0]
    edges = row_edge_strength(gray, x1, x2)

    # Work in a window below the initial guess
    y0 = int(max(top, bottom_guess - 0.03 * H))
    y1 = int(min(bottom_limit, bottom_guess + 0.30 * H))
    if y1 <= y0 + 10:
        return bottom_guess

    seg = edges[y0:y1]
    # Adaptive threshold: use a low percentile as "background edge level"
    bg_level = float(np.percentile(seg, 25))
    # Edge must be "meaningfully above background" to count as text/content
    text_thr = bg_level + 6.0

    # Helper: find first run where edges are low/high
    def find_run(start, condition_fn, min_run):
        y = start
        end = y1
        while y < end:
            if not condition_fn(y):
                y += 1
                continue
            s = y
            while y < end and condition_fn(y):
                y += 1
            e = y
            if (e - s) >= min_run:
                return s, e
        return None

    # 1) Find a "background padding" run near bottom_guess (low edges)
    low_cond = lambda yy: edges[yy] <= (bg_level + 2.0)
    padding = find_run(start=max(y0, bottom_guess - int(0.02 * H)),
                       condition_fn=low_cond,
                       min_run=max(10, int(0.01 * H)))
    if padding is None:
        # If we can't even identify padding, keep old bottom
        return bottom_guess

    padding_end = padding[1]

    # 2) Find "text/content" run after padding (high edges)
    high_cond = lambda yy: edges[yy] >= text_thr
    text_run = find_run(start=padding_end,
                        condition_fn=high_cond,
                        min_run=max(10, int(0.01 * H)))
    if text_run is None:
        # No text detected after padding
        return bottom_guess

    text_start = text_run[0]

    # 3) Find end of text: first sustained low-edge run after text_start
    # Allow some extra below detected text start
    search_start = int(min(text_start + int(0.04 * H), y1 - 1))
    end_run = find_run(start=search_start,
                       condition_fn=low_cond,
                       min_run=max(14, int(0.015 * H)))
    if end_run is None:
        # No clear end found; extend conservatively
        return min(bottom_limit, text_start + int(0.20 * H))

    text_end = end_run[0]
    return min(bottom_limit, text_end)


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

    # 1) Normalize
    img = resize_to_width(img, target_w=1080)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Theme
    theme = detect_theme(gray, w, h)

    # 3) Central width (avoid edges/icons)
    x1 = int(0.18 * w)
    x2 = int(0.82 * w)

    # 4) Base detection (same as your approach)
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
        raise HTTPException(status_code=422, detail="no_ad_detected")

    top, bottom_guess = gap

    # 5) Padding on top (keep)
    pad_top = int(0.01 * h)
    top = max(0, top - pad_top)

    # 6) Bottom: extend to include text block after image padding stripe
    bottom_limit = int(0.92 * h)  # avoid bottom navbar
    bottom = extend_bottom_to_include_text(gray, x1, x2, top, bottom_guess, bottom_limit)

    # final small padding
    bottom = min(bottom_limit, bottom + int(0.01 * h))

    crop = img[top:bottom, 0:w]
    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(content=out.tobytes(), media_type="image/png")
