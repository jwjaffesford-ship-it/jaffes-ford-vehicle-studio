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
    tyre_contact_darkness: int = 150,
    tyre_contact_size: int = 100,
) -> Image.Image:
    """Render soft projected car silhouette, broad chassis and focused wheel contacts.

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

    # Dense, correctly grounded tyre shadows: three DIFFERENT spatial scales.
    # Broad penumbra is soft, medium-scale tyre contact is defined, and the
    # small dark core attaches the rubber to the floor. These are ground
    # ellipses (not black patches on the body), controlled separately from the
    # diffuse whole-car shadow. All tyre points use the same coherent layers.
    contact_factor = (base / 110.0) * (tyre_contact_darkness / 100.0)
    footprint_scale = (tyre_contact_size / 100.0) * width_factor

    def contact_mask(
        rx_fraction: float, ry_fraction: float,
        alpha: float, sigma_x: float, sigma_y: float,
    ) -> np.ndarray:
        mask = np.zeros((bh, bw), dtype=np.uint8)
        for i, (x, y) in enumerate(tyre_points):
            # Extra, partly hidden wheels are fainter; never assume their
            # ground point is as certain as the two user-selected contacts.
            relative = 1.0 if i < 2 else 0.55
            rx = max(3, round(vw * rx_fraction * footprint_scale))
            ry = max(2, round(vw * ry_fraction * footprint_scale))
            cv2.ellipse(
                mask, (int(x), int(y)), (rx, ry), 0, 0, 360,
                min(238, round(alpha * contact_factor * relative)), -1,
            )
        return cv2.GaussianBlur(
            mask, (0, 0), sigmaX=max(0.7, sigma_x),
            sigmaY=max(0.7, sigma_y),
        )

    # Small halos make the floor feel continuous around the rubber.
    layers.append(contact_mask(
        0.059, 0.018, 46,
        max(4.0, blur * 0.44), max(2.0, blur * 0.20),
    ))
    # Mid-scale shadow is defined enough to reveal actual tyre contact.
    layers.append(contact_mask(
        0.034, 0.009, 94,
        max(2.0, blur * 0.16), max(1.0, blur * 0.075),
    ))
    # Darkest core sits very close to the bottom of each actual tyre.
    layers.append(contact_mask(
        0.018, 0.005, 118,
        max(1.0, blur * 0.065), max(0.8, blur * 0.035),
    ))

    # Charcoal-grey shadows are blended separately on the untouched background.
    for mask in layers:
        layer = Image.new("RGBA", bg.size, (19, 20, 24, 0))
        layer.putalpha(Image.fromarray(mask, mode="L"))
        bg = Image.alpha_composite(bg, layer)
    return bg
