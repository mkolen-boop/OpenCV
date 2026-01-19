from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, JSONResponse
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
    x1 = int(0.20 * w)
    x2 = int(0.80 * w)
    y1 = int(0.12 * h)
    y2 = int(0.28 * h)
    roi = gray[y1:y2, x1:x2]
    med = float(np.median(roi)) if roi.size else float(np.median(gray))
    return "light" if med >= 150.0 else "dark"


def merge_close_bands(bands, max_gap_px=10):
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


def find_bg_bands_color_edge(
    gray, x1, x2, theme,
    min_run,
    white_thr, white_frac,
    black_thr, black_frac,
    edge_thr, edge_low_frac
):
    roi = gray[:, x1:x2].astype(np.uint8)

    if theme == "light":
        frac_bg = (roi >= white_thr).mean(axis=1)
        is_bg_color = frac_bg >= white_frac
    else:
        frac_bg = (roi <= black_thr).mean(axis=1)
        is_bg_color = frac_bg >= black_frac

    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    low_edge_frac = (mag <= edge_thr).mean(axis=1)
    is_bg_edge = low_edge_frac >= edge_low_frac

    is_bg = is_bg_color & is_bg_edge

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


def find_bg_bands_mean_only(gray, x1, x2, theme, min_run=18, light_thr=245, dark_thr=12):
    """
    Найпростіший fallback: рядок фонового кольору за середнім значенням.
    Для light — близько до білого, для dark — близько до чорного.
    """
    roi = gray[:, x1:x2]
    row_mean = roi.mean(axis=1)

    if theme == "light":
        is_bg = row_mean >= light_thr
    else:
        is_bg = row_mean <= dark_thr

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


def pick_gap_skip_inner_stripes(bands, h, min_card_px, min_gap_px, min_bottom_band_px, prefer_lower_half=True):
    if len(bands) < 2:
        return None

    best = None  # (score, top, bottom)
    for i in range(len(bands) - 1):
        top = bands[i][1]
        bottom = bands[i + 1][0]
        gap = bottom - top
        bottom_band_h = bands[i + 1][1] - bands[i + 1][0]

        # ключова ідея: ігноримо внутрішні тонкі/ранні смужки
        if bottom < top + min_card_px:
            continue
        if gap < min_gap_px:
            continue
        if bottom_band_h < min_bottom_band_px:
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


def try_detect_gap(gray, w, h, theme, x1, x2):
    """
    3-ступеневий детект:
      A) строгий color+edge
      B) менш строгий color+edge
      C) fallback mean-only
    """
    # Common geometry
    min_card_px = int(0.22 * h)         # трохи м'якше, ніж 0.25*h
    min_gap_px = int(0.16 * h)          # м'якше
    min_bottom_band_px = int(0.018 * h) # м'якше

    # A) stricter
    bands = find_bg_bands_color_edge(
        gray, x1, x2, theme,
        min_run=18,
        white_thr=235, white_frac=0.990,
        black_thr=35,  black_frac=0.990,
        edge_thr=18.0, edge_low_frac=0.90
    )
    bands = merge_close_bands(bands, max_gap_px=10)
    gap = pick_gap_skip_inner_stripes(bands, h, min_card_px, min_gap_px, min_bottom_band_px)
    if gap is not None:
        return gap, {"stage": "A", "bands": len(bands)}

    # B) relaxed: дозволяємо більше “нефонового” та більше edge
    bands = find_bg_bands_color_edge(
        gray, x1, x2, theme,
        min_run=16,
        white_thr=232, white_frac=0.985,
        black_thr=40,  black_frac=0.985,
        edge_thr=22.0, edge_low_frac=0.86
    )
    bands = merge_close_bands(bands, max_gap_px=14)
    gap = pick_gap_skip_inner_stripes(bands, h, min_card_px, min_gap_px, min_bottom_band_px)
    if gap is not None:
        return gap, {"stage": "B", "bands": len(bands)}

    # C) fallback mean-only
    bands = find_bg_bands_mean_only(
        gray, x1, x2, theme,
        min_run=18,
        light_thr=246 if theme == "light" else 245,
        dark_thr=14
    )
    bands = merge_close_bands(bands, max_gap_px=12)
    gap = pick_gap_skip_inner_stripes(
        bands, h,
        min_card_px=int(0.20 * h),
        min_gap_px=int(0.14 * h),
        min_bottom_band_px=int(0.014 * h)
    )
    if gap is not None:
        return gap, {"stage": "C", "bands": len(bands)}

    return None, {"stage": "NONE", "bands": len(bands) if bands is not None else 0}


# -----------------------
# API
# -----------------------
@app.get("/health")
def health():
    return {"ok": True, "opencv_version": cv2.__version__}


@app.post("/crop-ad")
async def crop_ad(
    file: UploadFile = File(...),
    # якщо true — повертає JSON з debug-інфо замість картинки (для діагностики через n8n)
    debug: bool = Query(False),
    # якщо true — коли не знайшло, повертає 422 як раніше; якщо false — повертає оригінал (щоб n8n не падав)
    strict: bool = Query(False),
):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await file.read()
    img0 = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img0 is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    img = resize_to_width(img0, target_w=1080)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    theme = detect_theme(gray, w, h)

    # Центральна ширина (YouTube: аватар зліва, меню справа)
    x1 = int(0.18 * w)
    x2 = int(0.82 * w)

    gap, meta = try_detect_gap(gray, w, h, theme, x1, x2)

    if gap is None:
        if debug:
            return JSONResponse(
                status_code=200,
                content={"ok": False, "reason": "no_ad_detected", "theme": theme, **meta}
            )
        if strict:
            raise HTTPException(status_code=422, detail="no_ad_detected")

        # fallback: повертаємо оригінал, щоб n8n не валився
        ok, out0 = cv2.imencode(".png", img)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        return Response(
            content=out0.tobytes(),
            media_type="image/png",
            headers={"X-Crop-Status": "no_ad_detected", "X-Theme": theme}
        )

    top, bottom = gap

    # Padding + navbar protection
    pad_top = int(0.01 * h)
    pad_bottom = int(0.015 * h)
    top = max(0, top - pad_top)
    bottom = min(h, bottom + pad_bottom)
    bottom = min(bottom, int(0.92 * h))

    crop = img[top:bottom, 0:w]

    if debug:
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "theme": theme,
                "stage": meta.get("stage"),
                "bands": meta.get("bands"),
                "top": int(top),
                "bottom": int(bottom),
                "h": int(h),
                "w": int(w),
            }
        )

    ok, out = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return Response(
        content=out.tobytes(),
        media_type="image/png",
        headers={"X-Crop-Status": "cropped", "X-Theme": theme, "X-Stage": meta.get("stage", "")}
    )

