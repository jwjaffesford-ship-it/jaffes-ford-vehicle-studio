"""Jaffes Ford Vehicle Studio — first working prototype.

Upload a photo and the exact approved showroom background.  The app removes
background pixels from the photo (or honors an existing PNG alpha channel),
composites the unaltered vehicle pixels and a soft contact shadow, and exports
at the showroom background's native dimensions with 200-DPI metadata.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps

st.set_page_config(page_title="Jaffes Ford Vehicle Studio", page_icon="🚘", layout="wide")


@st.cache_data(show_spinner=False, max_entries=15)
def load_png(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        out = io.BytesIO()
        image.convert("RGBA").save(out, format="PNG")
        return out.getvalue()


@st.cache_resource(show_spinner=False)
def background_removal_session():
    from rembg import new_session
    return new_session("u2netp")


@st.cache_data(show_spinner=False, max_entries=10)
def generate_cutout(image_bytes: bytes) -> bytes:
    from rembg import remove
    result = remove(image_bytes, session=background_removal_session())
    with Image.open(io.BytesIO(result)) as image:
        out = io.BytesIO()
        image.convert("RGBA").save(out, format="PNG")
        return out.getvalue()


def parse_image(png_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(png_bytes)) as image:
        return image.convert("RGBA")


def crop_to_alpha(image: Image.Image) -> Image.Image:
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("No visible vehicle remains after background removal.")
    return image.crop(bbox)


def draw_contact_shadow(canvas: Image.Image, left: int, top: int, width: int, height: int,
                        opacity: int, softness: int, spread: float) -> Image.Image:
    if opacity <= 0:
        return canvas
    # This is a photographic approximation, not a physics-based ray tracer.
    shadow = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(shadow)
    feet = top + height
    half_w = max(10, int(width * 0.46 * spread))
    half_h = max(5, int(width * 0.052 * spread))
    mid = left + width // 2
    y = feet - max(2, int(height * 0.018))
    draw.ellipse((mid - half_w, y - half_h, mid + half_w, y + half_h), fill=opacity)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(0, int(softness))))
    shading = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shading.putalpha(shadow)
    return Image.alpha_composite(canvas, shading)


def render(background: Image.Image, vehicle: Image.Image, size_pct: int,
           x_pct: int, ground_pct: int, opacity: int, softness: int, spread: float):
    vehicle = crop_to_alpha(vehicle)
    bw, bh = background.size
    target_width = max(1, round(bw * size_pct / 100))
    target_height = max(1, round(target_width * vehicle.height / vehicle.width))
    vehicle = vehicle.resize((target_width, target_height), Image.Resampling.LANCZOS)
    left = round(bw * x_pct / 100 - target_width / 2)
    top = round(bh * ground_pct / 100 - target_height)

    composited = background.copy().convert("RGBA")
    composited = draw_contact_shadow(composited, left, top, target_width, target_height,
                                     opacity, softness, spread)
    # Alpha compositing correctly keeps translucent edges (unlike masking twice).
    vehicle_layer = Image.new("RGBA", composited.size, (0, 0, 0, 0))
    vehicle_layer.paste(vehicle, (left, top))
    composited = Image.alpha_composite(composited, vehicle_layer)
    fits = (left >= 0 and top >= 0 and left + target_width <= bw and top + target_height <= bh)
    return composited, fits


def crop_guide(image: Image.Image) -> Image.Image:
    preview = image.copy()
    w, h = preview.size
    # A portrait 4:5 region at full image height, centred horizontally.
    guide_width = min(w, round(h * 4 / 5))
    x = (w - guide_width) // 2
    draw = ImageDraw.Draw(preview)
    line_width = max(2, round(w / 700))
    draw.rectangle((x, 0, x + guide_width - 1, h - 1), outline=(220, 30, 40, 255), width=line_width)
    return preview


st.title("Jaffes Ford Vehicle Studio")
st.caption("Prototype 1 · Real image cut-out, fixed uploaded showroom background, contact shadows and full-size export")
st.info("Upload your **actual Ford showroom background**. This app never invents or replaces it.")

with st.sidebar:
    st.header("1 · Upload original images")
    background_upload = st.file_uploader("Approved Ford showroom background", type=["png", "jpg", "jpeg"], key="bg")
    vehicle_upload = st.file_uploader("Original vehicle photograph", type=["png", "jpg", "jpeg"], key="vehicle")
    remove_background = st.checkbox("Remove vehicle photo background automatically", value=True,
                                    help="Uncheck when uploading a vehicle PNG that is already transparent.")
    st.header("2 · Position vehicle")
    size_pct = st.slider("Vehicle width (% of showroom width)", 10, 105, 65)
    x_pct = st.slider("Horizontal centre (%)", 0, 100, 50)
    ground_pct = st.slider("Ground contact height (%)", 25, 100, 83)
    st.header("3 · Floor shadow")
    opacity = st.slider("Shadow strength", 0, 220, 115)
    softness = st.slider("Shadow softness (pixels)", 0, 90, 24)
    spread = st.slider("Shadow width", 0.5, 1.5, 1.0, 0.05)
    st.header("4 · Preview")
    show_crop = st.checkbox("Show centred 4:5 crop guide in preview", value=False)

if background_upload is None or vehicle_upload is None:
    st.write("Upload both images using the panel on the left to begin.")
    st.stop()

bg_bytes = load_png(background_upload.getvalue())
vehicle_bytes = load_png(vehicle_upload.getvalue())
background = parse_image(bg_bytes)
original_vehicle = parse_image(vehicle_bytes)

# Respect an already-cut-out PNG when the automatic option is turned off.
if remove_background:
    try:
        with st.spinner("Removing the original background from the vehicle photo..."):
            vehicle = parse_image(generate_cutout(vehicle_bytes))
    except Exception as ex:
        st.error("Automatic cut-out failed. Please retry or upload a transparent PNG vehicle image.")
        with st.expander("Technical details"):
            st.code(str(ex))
        st.stop()
else:
    vehicle = original_vehicle
    if vehicle.getchannel("A").getextrema()[0] == 255:
        st.warning("Your image has no transparent pixels. Use automatic background removal, or upload an already cut-out PNG.")

try:
    final, fits = render(background, vehicle, size_pct, x_pct, ground_pct,
                         opacity, softness, spread)
except ValueError as ex:
    st.error(str(ex))
    st.stop()

left_col, right_col = st.columns([1, 2])
with left_col:
    st.subheader("Vehicle cut-out")
    st.image(vehicle, use_container_width=True)
    st.caption("Review mirrors, wheels, glass and other fine edges. Automatic masking may need manual correction.")
with right_col:
    st.subheader("Showroom composite")
    st.image(crop_guide(final) if show_crop else final, use_container_width=True)
    st.caption("Preview guide is never included in the final exported photograph.")

if not fits:
    st.warning("Part of the vehicle is outside the showroom image. Adjust its size or position before exporting.")

output = io.BytesIO()
final.convert("RGB").save(output, format="PNG", dpi=(200, 200))
st.download_button("Download full-size PNG · 200 dpi", output.getvalue(),
                   file_name="Jaffes_Ford_vehicle_composite_200dpi.png", mime="image/png")
st.caption(f"Export size: {background.width} × {background.height} pixels. The original showroom pixel dimensions are preserved.")
st.divider()
st.markdown("**Coming in the next version:** locked showroom assets, all nine angle templates, automatic template placement, accurate roofline alignment, 4:5 safe-area validation and batch downloads.")
