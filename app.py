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


def tyre_shadow_points(vehicle: Image.Image, left: int, top: int,
                       tyre_fractions: tuple[float, float],
                       tyre_y_offsets: tuple[int, int]):
    """Return the two estimated tyre/floor contact points on the final canvas."""
    alpha = np.asarray(vehicle.getchannel("A"))
    points = []
    for fraction, y_offset in zip(tyre_fractions, tyre_y_offsets):
        x = left + round((vehicle.width - 1) * fraction)
        y = top + find_tyre_bottom(alpha, fraction) + y_offset
        points.append((x, y))
    return points


def draw_contact_shadow(canvas: Image.Image, vehicle: Image.Image, left: int, top: int,
                        opacity: int, softness: int, spread: float,
                        left_tyre_frac: float, right_tyre_frac: float,
                        left_tyre_y: int, right_tyre_y: int) -> Image.Image:
    """Use two independently grounded tyre shadows and a restrained ambient shadow.

    This is a photographic approximation: tyre centres can be adjusted in the UI.
    Shadow is applied *behind* vehicle pixels; the vehicle itself is not altered.
    """
    if opacity <= 0:
        return canvas
    width, height = vehicle.size
    points = tyre_shadow_points(
        vehicle, left, top, (left_tyre_frac, right_tyre_frac),
        (left_tyre_y, right_tyre_y),
    )

    # Faint shadow beneath the chassis. It follows the line between the tyres
    # rather than appearing as a large detached oval.
    ambient = Image.new("L", canvas.size, 0)
    ambient_draw = ImageDraw.Draw(ambient)
    ambient_draw.line(points, fill=min(255, round(opacity * 0.28)),
                      width=max(8, round(width * 0.055 * spread)))
    ambient = ambient.filter(ImageFilter.GaussianBlur(max(2, softness)))

    # More concentrated patches where the tyres meet the floor.
    contact = Image.new("L", canvas.size, 0)
    contact_draw = ImageDraw.Draw(contact)
    radius_x = max(5, round(width * 0.046 * spread))
    radius_y = max(3, round(width * 0.011 * spread))
    for x, y in points:
        contact_draw.ellipse((x - radius_x, y - radius_y, x + radius_x, y + radius_y),
                             fill=min(255, round(opacity * 1.25)))
    contact = contact.filter(ImageFilter.GaussianBlur(max(1, round(softness * 0.25))))

    # Alpha composite the dark layers against the original untouched showroom.
    for mask in (ambient, contact):
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.putalpha(mask)
        canvas = Image.alpha_composite(canvas, layer)
    return canvas


def render(background: Image.Image, vehicle: Image.Image, size_pct: int,
           x_pct: int, ground_pct: int, opacity: int, softness: int, spread: float,
           left_tyre_pct: int, right_tyre_pct: int,
           left_tyre_y: int = 0, right_tyre_y: int = 0,
           fine_x: int = 0, fine_y: int = 0, fine_scale: float = 1.0):
    vehicle = crop_to_alpha(vehicle)
    bw, bh = background.size
    target_width = max(1, round(bw * size_pct / 100 * fine_scale))
    target_height = max(1, round(target_width * vehicle.height / vehicle.width))
    vehicle = vehicle.resize((target_width, target_height), Image.Resampling.LANCZOS)
    left = round(bw * x_pct / 100 - target_width / 2) + fine_x
    top = round(bh * ground_pct / 100 - target_height) + fine_y

    composited = background.copy().convert("RGBA")
    composited = draw_contact_shadow(
        composited, vehicle, left, top, opacity, softness, spread,
        left_tyre_pct / 100, right_tyre_pct / 100,
        left_tyre_y, right_tyre_y
    )
    # Put original vehicle pixels above the shadows; preserve translucent edges.
    vehicle_layer = Image.new("RGBA", composited.size, (0, 0, 0, 0))
    vehicle_layer.paste(vehicle, (left, top))
    composited = Image.alpha_composite(composited, vehicle_layer)
    fits = (left >= 0 and top >= 0 and left + target_width <= bw and top + target_height <= bh)
    return composited, fits, (left, top, left + target_width, top + target_height), tuple(
        tyre_shadow_points(
            vehicle, left, top, (left_tyre_pct / 100, right_tyre_pct / 100),
            (left_tyre_y, right_tyre_y),
        )
    )


@st.cache_data(show_spinner=False, max_entries=6)
def estimate_template_guides(reference_png: bytes, bw: int, bh: int):
    """Find red roofline and crop-safe guides from the actual uploaded template."""
    ref = parse_image(reference_png).resize((bw, bh), Image.Resampling.LANCZOS).convert("RGB")
    rgb = np.asarray(ref, dtype=np.int16)
    red = (
        (rgb[:, :, 0] > 105)
        & (rgb[:, :, 0] > rgb[:, :, 1] + 34)
        & (rgb[:, :, 0] > rgb[:, :, 2] + 27)
    )
    # For the vertical guides, count red pixels across most of the preview height.
    vertical_counts = red[:, :].sum(axis=0)
    left_window = (max(0, round(bw * 0.07)), round(bw * 0.46))
    right_window = (round(bw * 0.54), min(bw, round(bw * 0.93)))
    left_slice = vertical_counts[left_window[0]:left_window[1]]
    right_slice = vertical_counts[right_window[0]:right_window[1]]
    safe_box = None
    if left_slice.size and right_slice.size:
        left_x = left_window[0] + int(np.argmax(left_slice))
        right_x = right_window[0] + int(np.argmax(right_slice))
        if (
            vertical_counts[left_x] >= bh * 0.28
            and vertical_counts[right_x] >= bh * 0.28
            and right_x - left_x >= bw * 0.35
        ):
            left_rows = np.flatnonzero(red[:, max(0, left_x - 2): min(bw, left_x + 3)].sum(axis=1) >= 2)
            right_rows = np.flatnonzero(red[:, max(0, right_x - 2): min(bw, right_x + 3)].sum(axis=1) >= 2)
            if left_rows.size and right_rows.size:
                top = max(int(np.percentile(left_rows, 3)), int(np.percentile(right_rows, 3)))
                bottom = min(int(np.percentile(left_rows, 97)), int(np.percentile(right_rows, 97)))
                if bottom - top >= bh * 0.5:
                    safe_box = (left_x, top, right_x, bottom)

    # The roofline is a mostly horizontal dashed red guide in the central image.
    y1, y2 = round(bh * 0.18), round(bh * 0.64)
    counts = red[y1:y2, round(bw * 0.1):round(bw * 0.9)].sum(axis=1)
    roof_y = None
    if counts.size and counts.max() >= bw * 0.13:
        roof_y = y1 + int(np.argmax(counts))
    return safe_box, roof_y


@st.cache_data(show_spinner=False, max_entries=6)
def estimate_template_vehicle_box(background_png: bytes, reference_png: bytes):
    """Suggest bounding box of the *reference* car by comparing it with the blank showroom.

    This is an estimate only. Colored guides, thin annotations and small artifacts
    are removed before component analysis. Results must be visually inspected.
    """
    bg = parse_image(background_png).convert("RGB")
    ref = parse_image(reference_png).convert("RGB").resize(bg.size, Image.Resampling.LANCZOS)
    bw, bh = bg.size
    sw = min(480, bw)
    sh = max(1, round(bh * sw / bw))
    bg_sm = np.asarray(bg.resize((sw, sh), Image.Resampling.BILINEAR), dtype=np.int16)
    ref_sm = np.asarray(ref.resize((sw, sh), Image.Resampling.BILINEAR), dtype=np.int16)
    diff = np.abs(bg_sm - ref_sm)
    strength = diff.mean(axis=2)
    # Ignore annotations of red or saturated blue. Those are not vehicle pixels.
    red = (
        (ref_sm[:, :, 0] > ref_sm[:, :, 1] + 30)
        & (ref_sm[:, :, 0] > ref_sm[:, :, 2] + 25)
        & (ref_sm[:, :, 0] > 100)
    )
    blue = (
        (ref_sm[:, :, 2] > ref_sm[:, :, 0] + 35)
        & (ref_sm[:, :, 2] > ref_sm[:, :, 1] + 20)
    )
    selected = (strength > 35) & (diff.max(axis=2) > 55) & ~(red | blue)
    # The car occupies the lower/middle region of these landscape templates.
    selected[:round(sh * 0.22), :] = False
    selected[round(sh * 0.97):, :] = False
    selected[:, :round(sw * 0.05)] = False
    selected[:, round(sw * 0.95):] = False
    mask = Image.fromarray((selected.astype(np.uint8) * 255), mode="L")
    mask = mask.filter(ImageFilter.MinFilter(3))
    mask = mask.filter(ImageFilter.MaxFilter(11))
    mask = mask.filter(ImageFilter.MinFilter(9))
    data = np.asarray(mask) > 120
    visited = np.zeros(data.shape, dtype=np.bool_)
    best = None
    best_score = 0.0
    # Iterative 8-neighbour flood fill on a thumbnail-sized mask.
    for sy, sx in zip(*np.where(data)):
        if visited[sy, sx]:
            continue
        visited[sy, sx] = True
        stack = [(int(sy), int(sx))]
        count = 0
        x0 = x1 = int(sx)
        y0 = y1 = int(sy)
        while stack:
            y, x = stack.pop()
            count += 1
            x0, x1 = min(x0, x), max(x1, x)
            y0, y1 = min(y0, y), max(y1, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1),
                           (y - 1, x - 1), (y - 1, x + 1), (y + 1, x - 1), (y + 1, x + 1)):
                if 0 <= nx < sw and 0 <= ny < sh and data[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if (
            w < sw * 0.16 or h < sh * 0.13
            or w > sw * 0.92 or h > sh * 0.66
            or count < sw * sh * 0.009
        ):
            continue
        centre_x = (x0 + x1) / (2 * sw)
        centre_y = (y0 + y1) / (2 * sh)
        location_weight = max(0.15, 1 - abs(centre_x - 0.5) * 0.6 - abs(centre_y - 0.6) * 0.5)
        score = count * location_weight
        if score > best_score:
            best_score = score
            best = (
                round(x0 * bw / sw), round(y0 * bh / sh),
                round((x1 + 1) * bw / sw), round((y1 + 1) * bh / sh),
            )
    return best


def proposed_placement(background: Image.Image, vehicle: Image.Image, bbox, roof_y,
                       safe_box, is_30_degree: bool):
    """Scale uniformly, centre on the reference car and align its top to roof guide."""
    bw, bh = background.size
    cutout = crop_to_alpha(vehicle)
    x0, y0, x1, y1 = bbox
    target_w = max(1, x1 - x0)
    if is_30_degree:
        if safe_box is not None:
            safe_left, safe_top, safe_right, safe_bottom = safe_box
        else:
            safe_width = min(bw, round(bh * 4 / 5))
            safe_left, safe_right = (bw - safe_width) // 2, (bw + safe_width) // 2
            safe_top, safe_bottom = 0, bh
        # Contain the whole car with small margins. Never stretch or change angle.
        target_w = min(target_w, max(1, int((safe_right - safe_left) * 0.96)))
    else:
        safe_left, safe_top, safe_right, safe_bottom = 0, 0, bw, bh
    size = max(10, min(105, round(target_w / bw * 100)))
    scaled_w = round(bw * size / 100)
    scaled_h = round(scaled_w * cutout.height / cutout.width)
    centre = (x0 + x1) / 2
    if is_30_degree:
        centre = min(safe_right - scaled_w / 2 - 1, max(safe_left + scaled_w / 2 + 1, centre))
    x_pct = max(0, min(100, round(centre * 100 / bw)))
    top = roof_y if roof_y is not None else y0
    top = max(safe_top, min(top, safe_bottom - scaled_h))
    ground_pct = max(25, min(100, round((top + scaled_h) * 100 / bh)))
    return size, x_pct, ground_pct


def apply_suggested_placement(suggestion):
    st.session_state["vehicle_width_slider"] = suggestion[0]
    st.session_state["vehicle_center_slider"] = suggestion[1]
    st.session_state["vehicle_ground_slider"] = suggestion[2]
    # Reset previous fine-tuning when applying a fresh reference suggestion.
    st.session_state["fine_x_slider"] = 0
    st.session_state["fine_y_slider"] = 0
    st.session_state["fine_scale_slider"] = 100.0


def draw_shadow_points(image: Image.Image, points) -> Image.Image:
    """Preview only: display exact coordinates where shadow anchor patches are placed."""
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    radius = max(5, round(image.width / 250))
    for i, (x, y) in enumerate(points, start=1):
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=(255, 200, 0, 215), outline=(0, 0, 0, 255), width=2)
        draw.text((x+radius+3, y-radius-3), f"Tyre {i}", fill=(0, 0, 0, 255),
                  stroke_width=2, stroke_fill=(255, 255, 255, 255))
    return overlay


def apply_front_30_shadow_preset():
    """A tested starting point for this Ranger 30-degree reference, not final QA."""
    st.session_state["tyre_front_pos"] = 53
    st.session_state["tyre_rear_pos"] = 93
    st.session_state["tyre_front_y"] = 0
    st.session_state["tyre_rear_y"] = 6
    st.session_state["shadow_marker_preview"] = True


def apply_crop_safe_settings(new_size_pct: int, new_x_pct: int, new_ground_pct: int,
                             new_scale: float, new_fine_x: int, new_fine_y: int):
    """Apply a safe-fit adjustment before the related Streamlit sliders are created."""
    st.session_state["vehicle_width_slider"] = new_size_pct
    st.session_state["vehicle_center_slider"] = new_x_pct
    st.session_state["vehicle_ground_slider"] = new_ground_pct
    st.session_state["fine_scale_slider"] = new_scale
    st.session_state["fine_x_slider"] = new_fine_x
    st.session_state["fine_y_slider"] = new_fine_y


def crop_safe_settings(background, vehicle_box, safe_box, roof_y,
                       size_pct, x_pct, fine_scale_pct):
    """Compute a conservative non-distorting 4:5 fit, using exact slider math.

    Maintain the roof guide whenever it can be reached without crossing the crop.
    If it cannot, prefer full crop containment. Nothing is applied automatically.
    """
    import math
    bw, bh = background.size
    sx0, sy0, sx1, sy1 = safe_box
    pad_x = max(6, round((sx1 - sx0) * 0.025))
    pad_y = max(6, round((sy1 - sy0) * 0.018))
    target_top = roof_y if roof_y is not None else vehicle_box[1]
    target_top = max(sy0 + pad_y, target_top)
    vw = vehicle_box[2] - vehicle_box[0]
    vh = vehicle_box[3] - vehicle_box[1]
    avail_w = sx1 - sx0 - 2 * pad_x
    avail_h = sy1 - pad_y - target_top
    if avail_w <= 0 or avail_h <= 0:
        return None
    factor = min(1.0, avail_w / max(1, vw), avail_h / max(1, vh))
    # Small guard accounts for rounding slider values back to image pixels.
    factor *= 0.996
    target_fine_scale = fine_scale_pct * factor
    new_size_pct = size_pct
    if target_fine_scale < 90.0:
        new_size_pct = max(10, math.floor(size_pct * factor))
        target_fine_scale = size_pct * fine_scale_pct * factor / new_size_pct
    new_scale = max(90.0, min(110.0, math.floor(target_fine_scale * 10) / 10))
    new_w = max(1, round(bw * new_size_pct / 100 * new_scale / 100))
    new_h = max(1, round(new_w * vh / max(1, vw)))
    if new_w > avail_w or new_h > avail_h:
        return None

    # Retain the current horizontal centre where possible, then clamp to safe edges.
    current_cx = (vehicle_box[0] + vehicle_box[2]) / 2
    cx = min(sx1 - pad_x - new_w / 2, max(sx0 + pad_x + new_w / 2, current_cx))
    new_x_pct = max(0, min(100, round(cx * 100 / bw)))
    new_fine_x = round(cx - bw * new_x_pct / 100)
    new_ground_pct = max(25, min(100, round((target_top + new_h) * 100 / bh)))
    new_fine_y = target_top - round(bh * new_ground_pct / 100 - new_h)
    if not (-100 <= new_fine_x <= 100 and -100 <= new_fine_y <= 100):
        return None
    return (new_size_pct, new_x_pct, new_ground_pct,
            new_scale, new_fine_x, new_fine_y)


def snap_roofline(new_fine_y: int):
    """Streamlit button callback, runs before widgets are instantiated."""
    st.session_state["fine_y_slider"] = new_fine_y


def draw_detected_guides(image: Image.Image, safe_box, roof_y) -> Image.Image:
    """Show the actual detected template roof and crop guides, preview only."""
    preview = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(preview)
    w, h = preview.size
    line_width = max(2, round(w / 750))
    if safe_box is not None:
        x0, y0, x1, y1 = safe_box
        draw.rectangle((x0, y0, x1, y1), outline=(235, 40, 40, 255), width=line_width)
    if roof_y is not None:
        dash = max(8, round(w * 0.016))
        for x in range(0, w, dash * 2):
            draw.line((x, roof_y, min(w - 1, x + dash), roof_y), fill=(235, 40, 40, 255), width=line_width)
    return preview


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
st.caption("Prototype 1.8 · Safer 4:5 crop fitting, roofline alignment and tyre-contact shadows")
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
    show_template_guides = st.checkbox(
        "Show roofline and crop guides (without ghost vehicle)", value=True,
        help="These guides are detected from the uploaded Ford angle template and appear only in the preview.",
    )
    st.caption("The reference can suggest placement, but final fit must still be checked.")
    st.header("3 · Position vehicle")
    size_pct = st.slider("Vehicle width (% of showroom width)", 10, 105, 65, key="vehicle_width_slider")
    x_pct = st.slider("Horizontal centre (%)", 0, 100, 50, key="vehicle_center_slider")
    ground_pct = st.slider("Ground contact height (%)", 25, 100, 83, key="vehicle_ground_slider")
    with st.expander("Fine positioning (pixels)"):
        st.caption("Make small adjustments after applying the suggested placement. Zero means no fine adjustment.")
        fine_x = st.slider("Fine horizontal shift (pixels)", -100, 100, 0, 1, key="fine_x_slider")
        fine_y = st.slider("Fine vertical shift (pixels)", -100, 100, 0, 1, key="fine_y_slider")
        fine_scale_pct = st.slider("Fine size adjustment (%)", 90.0, 110.0, 100.0, 0.1, key="fine_scale_slider")
    st.header("4 · Floor shadow")
    opacity = st.slider("Shadow strength", 0, 220, 80)
    softness = st.slider("Shadow softness (pixels)", 0, 90, 18)
    spread = st.slider("Shadow width", 0.5, 1.5, 0.85, 0.05)
    with st.expander("Tyre contact positions (optional)"):
        st.caption("Move the two contact points if a shadow misses a tyre.")
        left_tyre_pct = st.slider("Left tyre position (% of cut-out width)", 5, 95, 50, key="tyre_front_pos")
        right_tyre_pct = st.slider("Right tyre position (% of cut-out width)", 5, 95, 87, key="tyre_rear_pos")
        left_tyre_y = st.slider("Tyre 1 shadow vertical offset (pixels)", -50, 50, 0, 1, key="tyre_front_y")
        right_tyre_y = st.slider("Tyre 2 shadow vertical offset (pixels)", -50, 50, 0, 1, key="tyre_rear_y")
        show_shadow_points = st.checkbox("Show tyre-shadow points on preview", value=False, key="shadow_marker_preview")
        if angle_label == "Front 30° — view 2":
            st.button(
                "Apply Ranger 30° shadow starting points",
                on_click=apply_front_30_shadow_preset,
                help="Based on the front and rear wheel positions in the tested grey Ranger image. Fine-tune for other vehicles.",
            )
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
    final, fits, vehicle_box, shadow_points = render(background, vehicle, size_pct, x_pct, ground_pct,
                         opacity, softness, spread, left_tyre_pct, right_tyre_pct,
                         left_tyre_y, right_tyre_y,
                         fine_x=fine_x, fine_y=fine_y, fine_scale=fine_scale_pct / 100)
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
    if reference is not None and show_template_guides:
        detected_safe_box, detected_roof_line = estimate_template_guides(
            template_bytes, background.width, background.height
        )
        preview = draw_detected_guides(preview, detected_safe_box, detected_roof_line)
    if show_shadow_points:
        preview = draw_shadow_points(preview, shadow_points)
    st.image(preview, use_container_width=True)
    st.caption("Reference/template and red crop guides are preview-only. They are NEVER included in the final download.")

if reference is not None:
    guide_box_for_check, roof_line_for_check = estimate_template_guides(
        template_bytes, background.width, background.height
    )
    if roof_line_for_check is not None:
        roof_gap_px = vehicle_box[1] - roof_line_for_check
        if abs(roof_gap_px) <= 2:
            st.success("Roofline alignment: within 2 pixels of the template guide.")
        elif roof_gap_px > 0:
            st.info(
                f"The top of the vehicle is about {roof_gap_px} pixels BELOW the roof-level guide."
            )
        else:
            st.warning(
                f"The top of the vehicle is about {-roof_gap_px} pixels ABOVE the roof-level guide."
            )
        target_fine_y = fine_y - roof_gap_px
        # Moving the vehicle must not violate the template 4:5 crop-safe area.
        within_fine_slider = -100 <= target_fine_y <= 100
        moved_box = (
            vehicle_box[0], vehicle_box[1] - roof_gap_px,
            vehicle_box[2], vehicle_box[3] - roof_gap_px,
        )
        if "30°" in angle_label:
            if guide_box_for_check is not None:
                sx0, sy0, sx1, sy1 = guide_box_for_check
            else:
                swidth = min(background.width, round(background.height * 4 / 5))
                sx0 = (background.width - swidth) // 2
                sy0 = 0
                sx1 = sx0 + swidth
                sy1 = background.height
        else:
            sx0, sy0, sx1, sy1 = 0, 0, background.width, background.height
        crop_safe = (
            moved_box[0] >= sx0 and moved_box[2] <= sx1
            and moved_box[1] >= sy0 and moved_box[3] <= sy1
        )
        if within_fine_slider and crop_safe and abs(roof_gap_px) > 2:
            st.button(
                f"Snap vehicle to roofline ({roof_gap_px:+d} px correction)",
                on_click=snap_roofline,
                args=(target_fine_y,),
                type="primary",
                help="Moves the original vehicle vertically without scaling, rotating or altering it.",
            )
        elif abs(roof_gap_px) > 2:
            st.caption(
                "Automatic roofline snap is unavailable because it would cross the crop boundary "
                "or exceed the fine-position range. Adjust vehicle size or position first."
            )
    with st.expander("One-click placement suggestion", expanded=True):
        try:
            with st.spinner("Comparing the angle reference with the empty showroom..."):
                guide_box, roof_line = estimate_template_guides(template_bytes, background.width, background.height)
                target_box = estimate_template_vehicle_box(bg_bytes, template_bytes)
            if target_box is None:
                st.warning(
                    "I couldn't confidently distinguish the reference vehicle from the template markings. "
                    "You can still use the translucent overlay and the manual position sliders."
                )
            else:
                suggestion = proposed_placement(
                    background, vehicle, target_box, roof_line, guide_box,
                    "30°" in angle_label,
                )
                st.write(
                    "Reference-based estimate: width **{}%**, horizontal centre **{}%**, "
                    "ground contact **{}%**.".format(*suggestion)
                )
                st.button(
                    "Apply suggested size and position",
                    on_click=apply_suggested_placement,
                    args=(suggestion,),
                    type="primary",
                    help="Estimation only. Check the roofline, wheels and 4:5 boundaries afterward.",
                )
                st.caption(
                    "The program estimates the truck's bounding box from your angle template. "
                    "It's not yet a pixel-perfect alignment system: inspect the result and fine-tune if needed."
                )
        except Exception as exc:
            st.warning("Automatic placement estimation was not possible for this template. Manual controls are still available.")
            with st.expander("Placement diagnostic"):
                st.code(str(exc))
    with st.expander("Show selected Ford angle template on its own"):
        st.image(reference, use_container_width=True)
        st.caption("Compare roofline, wheel position and vehicle proportions with the reference.")
else:
    st.info("No Ford angle template uploaded yet. Upload your approved template ZIP, or one angle image, to compare vehicle placement.")

if "30°" in angle_label:
    width, height = background.size
    if reference is not None:
        selected_guide, selected_roof = estimate_template_guides(
            template_bytes, width, height
        )
    else:
        selected_guide, selected_roof = None, None
    if selected_guide is None:
        safe_width = min(width, round(height * 4 / 5))
        safe_left = (width - safe_width) // 2
        safe_top = 0
        safe_right = safe_left + safe_width
        safe_bottom = height
    else:
        safe_left, safe_top, safe_right, safe_bottom = selected_guide

    padding_x = max(6, round((safe_right - safe_left) * 0.025))
    padding_y = max(6, round((safe_bottom - safe_top) * 0.018))
    crop_fit = (
        vehicle_box[0] >= safe_left and vehicle_box[2] <= safe_right
        and vehicle_box[1] >= safe_top and vehicle_box[3] <= safe_bottom
    )
    padded_fit = (
        vehicle_box[0] >= safe_left + padding_x
        and vehicle_box[2] <= safe_right - padding_x
        and vehicle_box[1] >= safe_top + padding_y
        and vehicle_box[3] <= safe_bottom - padding_y
    )
    if padded_fit:
        st.success("Vehicle inside the detected 4:5 crop guide with a 2.5% side safety margin.")
    elif crop_fit:
        st.warning(
            "The entire vehicle fits within the 4:5 crop area, but the mirrors or bodywork "
            "are close to the edges. A small safety margin is recommended."
        )
    else:
        st.warning(
            "Some of the vehicle extends beyond the detected 4:5 crop area. "
            "Adjust the size or position before using this image."
        )
    if not padded_fit:
        safe_settings = crop_safe_settings(
            background, vehicle_box,
            (safe_left, safe_top, safe_right, safe_bottom),
            selected_roof, size_pct, x_pct, fine_scale_pct,
        )
        if safe_settings is not None:
            st.button(
                "Fit vehicle inside 4:5 crop with safe margins",
                on_click=apply_crop_safe_settings,
                args=safe_settings,
                type="primary",
                help="Reduces size proportionally and adjusts position to preserve the roofline where possible. Does not alter the vehicle image or background.",
            )
        else:
            st.caption(
                "Automatic safe-fit is unavailable for the current constraints. "
                "Reduce vehicle width slightly or adjust the position sliders."
            )

if not fits:
    st.warning("Part of the vehicle is outside the showroom image. Adjust its size or position before exporting.")

output = io.BytesIO()
final.convert("RGB").save(output, format="PNG", dpi=(200, 200))
st.download_button("Download full-size PNG · 200 dpi", output.getvalue(),
                   file_name="Jaffes_Ford_vehicle_composite_200dpi.png", mime="image/png")
st.caption(f"Export size: {background.width} × {background.height} pixels. The original showroom pixel dimensions are preserved.")
st.divider()
st.markdown("**Next upgrades:** locked showroom and template references, model-specific alignment presets, manual mask touch-up and batch downloads.")
