"""Jaffes Ford Vehicle Studio — first working prototype.

Upload a photo and the exact approved showroom background.  The app removes
background pixels from the photo (or honors an existing PNG alpha channel),
composites the unaltered vehicle pixels and a soft contact shadow, and exports
at the showroom background's native dimensions with 200-DPI metadata.
"""
from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps

st.set_page_config(page_title="Jaffes Ford Vehicle Studio", page_icon="🚘", layout="wide")

# The ZIP supplied by the dealership uses these nine numbered template files.
# The ZIP is read in memory; nothing is uploaded to the public GitHub repository.
TEMPLATE_ANGLES = [
    ("02", "Left side"),
    ("03", "Right side"),
    ("04", "Front 30° — view 1"),
    ("05", "Front 30° — view 2"),
    ("06", "Straight front"),
    ("07", "Straight rear"),
    ("08", "Rear 30° — view 1"),
    ("09", "Rear 30° — view 2"),
    ("10", "Rear tailgate open"),
]


def template_from_zip(data: bytes, prefix: str):
    """Return a PNG/JPEG angle reference from the approved numbered ZIP."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            candidates = [
                n for n in archive.namelist()
                if n.split("/")[-1].lower().startswith(prefix + "_")
                and n.lower().endswith((".png", ".jpg", ".jpeg"))
                and not n.startswith("__MACOSX")
            ]
            if not candidates:
                return None
            return load_png(archive.read(sorted(candidates)[0]))
    except (zipfile.BadZipFile, OSError):
        return None


def preview_with_template(final: Image.Image, reference: Image.Image, opacity: int) -> Image.Image:
    """Ghost the angle reference over the preview only, never over the final PNG."""
    size = final.size
    reference = reference.resize(size, Image.Resampling.LANCZOS)
    base = final.convert("RGBA")
    alpha = reference.getchannel("A")
    factor = opacity / 100
    alpha = alpha.point(lambda n: round(n * factor))
    ghost = reference.copy()
    ghost.putalpha(alpha)
    return Image.alpha_composite(base, ghost)



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
    return composited, fits, (left, top, left + target_width, top + target_height)


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
st.caption("Prototype 1.3 · Template comparison, refined masking, tyre shadows and clean 200-dpi export")
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
    st.header("2 · Ford angle template")
    angle_label = st.selectbox("Reference angle", [name for _, name in TEMPLATE_ANGLES])
    angle_prefix = next(prefix for prefix, name in TEMPLATE_ANGLES if name == angle_label)
    template_zip = st.file_uploader(
        "Upload complete approved Ford template ZIP (optional)", type=["zip"], key="templates_zip",
        help="The nine approved angle files must retain their original numbered filenames.",
    )
    angle_reference = st.file_uploader(
        "Or upload just this angle template", type=["png", "jpg", "jpeg"], key="angle_reference",
    )
    show_template = st.checkbox("Ghost reference over preview", value=True)
    template_opacity = st.slider("Reference visibility (%)", 10, 80, 40, 5)
    st.caption("These references help manual matching. This version does not yet auto-align the vehicle.")
    st.header("3 · Position vehicle")
    size_pct = st.slider("Vehicle width (% of showroom width)", 10, 105, 65)
    x_pct = st.slider("Horizontal centre (%)", 0, 100, 50)
    ground_pct = st.slider("Ground contact height (%)", 25, 100, 83)
    st.header("4 · Floor shadow")
    opacity = st.slider("Shadow strength", 0, 220, 80)
    softness = st.slider("Shadow softness (pixels)", 0, 90, 18)
    spread = st.slider("Shadow width", 0.5, 1.5, 0.85, 0.05)
    with st.expander("Tyre contact positions (optional)"):
        st.caption("Move the two contact points if a shadow misses a tyre.")
        left_tyre_pct = st.slider("Left tyre position (% of cut-out width)", 5, 95, 50)
        right_tyre_pct = st.slider("Right tyre position (% of cut-out width)", 5, 95, 87)
    st.header("5 · Preview")
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
    final, fits, vehicle_box = render(background, vehicle, size_pct, x_pct, ground_pct,
                         opacity, softness, spread, left_tyre_pct, right_tyre_pct)
except ValueError as ex:
    st.error(str(ex))
    st.stop()

template_bytes = None
if angle_reference is not None:
    template_bytes = load_png(angle_reference.getvalue())
elif template_zip is not None:
    template_bytes = template_from_zip(template_zip.getvalue(), angle_prefix)
if template_zip is not None and template_bytes is None:
    st.warning("The selected angle was not found in the ZIP. Please choose a matching individual template image.")

reference = parse_image(template_bytes) if template_bytes else None

left_col, right_col = st.columns([1, 2])
with left_col:
    st.subheader("Vehicle cut-out")
    st.image(vehicle, use_container_width=True)
    st.caption("Review above the roof, mirrors, wheel edges and any remaining source floor. Increase 'Remove background haze' or select the Detailed model if needed.")
with right_col:
    st.subheader("Showroom composite")
    preview = crop_guide(final) if show_crop else final
    if reference is not None and show_template:
        preview = preview_with_template(preview, reference, template_opacity)
    st.image(preview, use_container_width=True)
    st.caption("Reference/template and red crop guides are preview-only. They are NEVER included in the final download.")

if reference is not None:
    with st.expander("Show selected Ford angle template on its own"):
        st.image(reference, use_container_width=True)
        st.caption("Compare roofline, wheel position and vehicle proportions with the reference.")
else:
    st.info("No Ford angle template uploaded yet. Upload your approved template ZIP, or one angle image, to compare vehicle placement.")

if "30°" in angle_label:
    width, height = background.size
    safe_width = min(width, round(height * 4 / 5))
    safe_left = (width - safe_width) // 2
    safe_right = safe_left + safe_width
    crop_fit = vehicle_box[0] >= safe_left and vehicle_box[2] <= safe_right and vehicle_box[1] >= 0 and vehicle_box[3] <= height
    if crop_fit:
        st.success("The vehicle appears inside the centred 4:5 crop guide. Check the actual angle template, too.")
    else:
        st.warning("The vehicle extends outside the centred 4:5 crop guide. Decrease width or adjust position before exporting.")

if not fits:
    st.warning("Part of the vehicle is outside the showroom image. Adjust its size or position before exporting.")

output = io.BytesIO()
final.convert("RGB").save(output, format="PNG", dpi=(200, 200))
st.download_button("Download full-size PNG · 200 dpi", output.getvalue(),
                   file_name="Jaffes_Ford_vehicle_composite_200dpi.png", mime="image/png")
st.caption(f"Export size: {background.width} × {background.height} pixels. The original showroom pixel dimensions are preserved.")
st.divider()
st.markdown("**Next upgrades:** persistent locked templates, automatic reference alignment, roofline matching, exact crop-safe enforcement, improved tyre grounding, and batch downloads.")
