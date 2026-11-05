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


@st.cache_resource(show_spinner=False, max_entries=1)
def background_removal_session(model_name: str):
    from rembg import new_session
    return new_session(model_name)


@st.cache_data(show_spinner=False, max_entries=5)
def generate_cutout(image_bytes: bytes, model_name: str) -> bytes:
    from rembg import remove
    # Rembg's built-in mask post-processing reduces noise and stray islands.
    result = remove(
        image_bytes,
        session=background_removal_session(model_name),
        post_process_mask=True,
    )
    with Image.open(io.BytesIO(result)) as image:
        out = io.BytesIO()
        image.convert("RGBA").save(out, format="PNG")
        return out.getvalue()


def refine_cutout(image: Image.Image, cleanup: int) -> Image.Image:
    """Remove weakly selected background haze while preserving vehicle RGB pixels.

    The slider does not regenerate or recolor the vehicle: it changes only alpha.
    Strong cleanup may erase genuine dark or translucent details, so review edges.
    """
    if cleanup <= 0:
        return image
    result = image.copy()
    alpha = np.asarray(image.getchannel("A"), dtype=np.float32)
    threshold = min(190, round(cleanup * 1.8))
    alpha = np.clip((alpha - threshold) * (255.0 / (255 - threshold)), 0, 255)
    result.putalpha(Image.fromarray(alpha.astype(np.uint8), mode="L"))
    return result


def parse_image(png_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(png_bytes)) as image:
        return image.convert("RGBA")


def crop_to_alpha(image: Image.Image) -> Image.Image:
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("No visible vehicle remains after background removal.")
    return image.crop(bbox)


def find_tyre_bottom(alpha: np.ndarray, fraction: float) -> int:
    """Estimate the lower outline near a chosen wheel, in cut-out pixels."""
    height, width = alpha.shape
    centre = min(width - 1, max(0, round((width - 1) * fraction)))
    radius = max(3, round(width * 0.035))
    region = alpha[:, max(0, centre - radius): min(width, centre + radius + 1)]
    # Sampling the bottommost opaque point in each column is more reliable
    # than finding the lowest alpha pixel anywhere (which can be stray noise).
    bottoms = []
    for col in range(region.shape[1]):
        positions = np.flatnonzero(region[:, col] > 140)
        if positions.size:
            bottoms.append(int(positions[-1]))
    return int(np.percentile(bottoms, 80)) if bottoms else round(height * 0.9)


def draw_contact_shadow(canvas: Image.Image, vehicle: Image.Image, left: int, top: int,
                        opacity: int, softness: int, spread: float,
                        left_tyre_frac: float, right_tyre_frac: float) -> Image.Image:
    """Use two independently grounded tyre shadows and a restrained ambient shadow.

    This is a photographic approximation: tyre centres can be adjusted in the UI.
    Shadow is applied *behind* vehicle pixels; the vehicle itself is not altered.
    """
    if opacity <= 0:
        return canvas
    width, height = vehicle.size
    alpha = np.asarray(vehicle.getchannel("A"))
    points = []
    for frac in (left_tyre_frac, right_tyre_frac):
        x = left + round((width - 1) * frac)
        y = top + find_tyre_bottom(alpha, frac)
        points.append((x, y))

    # Faint shadow beneath the chassis. It follows the line between the tyres
    # rather than appearing as a large detached oval.
    ambient = Image.new("L", canvas.size, 0)
    ambient_draw = ImageDraw.Draw(ambient)
    ambient_draw.line(points, fill=min(255, round(opacity * 0.4)),
                      width=max(8, round(width * 0.055 * spread)))
    ambient = ambient.filter(ImageFilter.GaussianBlur(max(2, softness)))

    # More concentrated patches where the tyres meet the floor.
    contact = Image.new("L", canvas.size, 0)
    contact_draw = ImageDraw.Draw(contact)
    radius_x = max(5, round(width * 0.055 * spread))
    radius_y = max(3, round(width * 0.012 * spread))
    for x, y in points:
        contact_draw.ellipse((x - radius_x, y - radius_y, x + radius_x, y + radius_y),
                             fill=min(255, round(opacity * 0.9)))
    contact = contact.filter(ImageFilter.GaussianBlur(max(1, round(softness * 0.32))))

    # Alpha composite the dark layers against the original untouched showroom.
    for mask in (ambient, contact):
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.putalpha(mask)
        canvas = Image.alpha_composite(canvas, layer)
    return canvas


def render(background: Image.Image, vehicle: Image.Image, size_pct: int,
           x_pct: int, ground_pct: int, opacity: int, softness: int, spread: float,
           left_tyre_pct: int, right_tyre_pct: int):
    vehicle = crop_to_alpha(vehicle)
    bw, bh = background.size
    target_width = max(1, round(bw * size_pct / 100))
    target_height = max(1, round(target_width * vehicle.height / vehicle.width))
    vehicle = vehicle.resize((target_width, target_height), Image.Resampling.LANCZOS)
    left = round(bw * x_pct / 100 - target_width / 2)
    top = round(bh * ground_pct / 100 - target_height)

    composited = background.copy().convert("RGBA")
    composited = draw_contact_shadow(
        composited, vehicle, left, top, opacity, softness, spread,
        left_tyre_pct / 100, right_tyre_pct / 100
    )
    # Put original vehicle pixels above the shadows; preserve translucent edges.
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
st.caption("Prototype 1.2 · Refined background masking, tyre contact shadows and full-size export")
st.info("Upload your **actual Ford showroom background**. This app never invents or replaces it.")

with st.sidebar:
    st.header("1 · Upload original images")
    background_upload = st.file_uploader("Approved Ford showroom background", type=["png", "jpg", "jpeg"], key="bg")
    vehicle_upload = st.file_uploader("Original vehicle photograph", type=["png", "jpg", "jpeg"], key="vehicle")
    remove_background = st.checkbox("Remove vehicle photo background automatically", value=True,
                                    help="Uncheck when uploading a vehicle PNG that is already transparent.")
    if remove_background:
        model_choice = st.selectbox(
            "Background-removal quality",
            ("Standard (faster)", "Detailed (more accurate, uses more memory)"),
            index=0,
            help="Try Detailed if the standard cut-out leaves pieces of the original scenery. The detailed model may exceed free hosting resources.",
        )
        cleanup = st.slider(
            "Remove background haze",
            min_value=0, max_value=90, value=35, step=5,
            help="Higher values remove faint unwanted pixels, but may also trim thin or translucent vehicle details. Review roof, mirrors, tyres and glass.",
        )
    else:
        model_choice = "Standard (faster)"
        cleanup = 0
    st.header("2 · Position vehicle")
    size_pct = st.slider("Vehicle width (% of showroom width)", 10, 105, 65)
    x_pct = st.slider("Horizontal centre (%)", 0, 100, 50)
    ground_pct = st.slider("Ground contact height (%)", 25, 100, 83)
    st.header("3 · Floor shadow")
    opacity = st.slider("Shadow strength", 0, 220, 80)
    softness = st.slider("Shadow softness (pixels)", 0, 90, 18)
    spread = st.slider("Shadow width", 0.5, 1.5, 0.85, 0.05)
    with st.expander("Tyre contact positions (optional)"):
        st.caption("Move the two contact points if a shadow misses a tyre.")
        left_tyre_pct = st.slider("Left tyre position (% of cut-out width)", 5, 95, 50)
        right_tyre_pct = st.slider("Right tyre position (% of cut-out width)", 5, 95, 87)
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
            selected_model = "isnet-general-use" if model_choice.startswith("Detailed") else "u2netp"
            vehicle = refine_cutout(parse_image(generate_cutout(vehicle_bytes, selected_model)), cleanup)
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
                         opacity, softness, spread, left_tyre_pct, right_tyre_pct)
except ValueError as ex:
    st.error(str(ex))
    st.stop()

left_col, right_col = st.columns([1, 2])
with left_col:
    st.subheader("Vehicle cut-out")
    st.image(vehicle, use_container_width=True)
    st.caption("Review above the roof, mirrors, wheel edges and any remaining source floor. Increase 'Remove background haze' or select the Detailed model if needed.")
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
