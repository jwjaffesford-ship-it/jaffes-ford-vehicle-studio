"""Perspective-aware photographic shadows, with original vehicle pixels preserved."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def render_showroom_shadows(
    background: Image.Image,
    vehicle: Image.Image,
    left: int,
    top: int,
    tyre_points: list[tuple[int, int]],
    *,
    strength: int,
    softness: int,
    width_factor: float,
    underbody_strength: int,
    underbody_depth: int,
    cast_strength: int = 60,
    cast_horizontal: int = -35,
    cast_vertical: int = -35,
) -> Image.Image:
    """Render soft projected vehicle silhouette + broad underbody + tyre contacts.

    Uses a 2.5D perspective approximation, not physical 3D lighting. Shadows
    are placed under the original cut-out without modifying showroom or car RGB.
    """
    if strength <= 0:
        return background.copy().convert("RGBA")
    bg = background.copy().convert("RGBA")
    bw, bh = bg.size
    vw, vh = vehicle.size
    if vw <= 1 or vh <= 1 or len(tyre_points) < 2:
        return bg

    (xnear, ynear), (xrear, yrear) = tyre_points[:2]
    body_front_x = left + round(vw * 0.07)
    body_rear_x = left + round(vw * 0.95)
    body_front_y = ynear - round(vh * 0.12)
    body_rear_y = yrear - round(vh * 0.07)
    back_front_y = ynear - round(vh * 0.25)
    back_rear_y = yrear - round(vh * 0.16)

    shift_x = cast_horizontal * (vw / 750.0)
    shift_y = cast_vertical * (vw / 750.0)
    depth_factor = underbody_depth / 100.0
    blur = max(2, int(softness))
    base = min(220, max(0, strength))
    layers = []

    if cast_strength > 0:
        alpha = np.asarray(vehicle.getchannel("A"), dtype=np.uint8)
        # Project only the alpha silhouette onto a quadrilateral on the floor.
        # Bottom remains near the floor contact; roof is cast farther away.
        src = np.float32([[0, 0], [vw - 1, 0], [vw - 1, vh - 1], [0, vh - 1]])
        dst = np.float32([
            [body_front_x + shift_x, back_front_y + shift_y * depth_factor],
            [body_rear_x + shift_x, back_rear_y + shift_y * depth_factor],
            [body_rear_x + shift_x * 0.12, body_rear_y],
            [body_front_x + shift_x * 0.12, body_front_y],
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        projected = cv2.warpPerspective(
            alpha, matrix, (bw, bh), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        projected = cv2.GaussianBlur(
            projected, (0, 0), sigmaX=max(6.0, blur * 1.5),
            sigmaY=max(6.0, blur * 1.2),
        )
        cast = np.clip(
            projected.astype(np.float32) * (base / 200.0)
            * (cast_strength / 100.0) * 0.72, 0, 115,
        ).astype(np.uint8)
        layers.append(cast)

    if underbody_strength > 0:
        # Broad polygon covers both front and rear underside, not just a line
        # connecting two tyres.
        under = np.zeros((bh, bw), dtype=np.uint8)
        depth = round(vw * 0.07 * width_factor * depth_factor)
        polygon = np.array([
            [body_front_x, body_front_y],
            [xnear, ynear + round(vh * 0.014)],
            [xrear, yrear + round(vh * 0.010)],
            [body_rear_x, body_rear_y - depth],
            [left + round(vw * 0.64), yrear - depth - round(vh * 0.05)],
            [left + round(vw * 0.20), back_front_y - round(depth * 0.55)],
        ], dtype=np.int32)
        cv2.fillPoly(
            under, [polygon],
            min(180, round(base * (underbody_strength / 100.0) * 0.75)),
        )
        under = cv2.GaussianBlur(
            under, (0, 0), sigmaX=max(5.0, blur * 1.25),
            sigmaY=max(4.0, blur * 0.85),
        )
        layers.append(under)

    for i, (x, y) in enumerate(tyre_points):
        contact = np.zeros((bh, bw), dtype=np.uint8)
        wheel_strength = 1.0 if i < 2 else 0.65
        rx = max(6, round(vw * 0.043 * width_factor))
        ry = max(3, round(vw * 0.010 * width_factor))
        cv2.ellipse(
            contact, (int(x), int(y)), (rx, ry), 0, 0, 360,
            min(230, round(base * 1.15 * wheel_strength)), -1,
        )
        contact = cv2.GaussianBlur(
            contact, (0, 0), sigmaX=max(2.0, blur * 0.28),
            sigmaY=max(1.0, blur * 0.18),
        )
        layers.append(contact)

    # Charcoal-grey shadows are blended separately on the untouched background.
    for mask in layers:
        layer = Image.new("RGBA", bg.size, (19, 20, 24, 0))
        layer.putalpha(Image.fromarray(mask, mode="L"))
        bg = Image.alpha_composite(bg, layer)
    return bg
