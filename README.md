# Jaffes Ford Vehicle Studio — Prototype 1

A real Python/Streamlit application to cut out vehicle photos, place original vehicle pixels on an **unchanged, user-supplied showroom image**, add a soft floor contact shadow, adjust placement and download the full-size PNG with 200-DPI metadata.

## What you need to do

You do **not** need to create the files manually: `app.py` and `requirements.txt` are already included.

To make it accessible from a web browser:

1. Unzip this starter project on your computer.
2. Upload `app.py` and `requirements.txt` into a **new GitHub repository** (a GitHub account will be required).
3. Go to https://share.streamlit.io and sign in using GitHub.
4. Choose **Create app** and select your repository, branch `main` and main file `app.py`.
5. Deploy. The first run can take time to install dependencies and download the open-source image segmentation model.
6. Upload your **approved Ford showroom background** and a vehicle photo to the application.

You may also run this locally if you already have Python installed:

```
python -m pip install -r requirements.txt
streamlit run app.py
```

## Current scope and limitations

- This is the first testable version, **not the finished nine-template Ford production system**.
- It needs your real background and vehicle photographs, uploaded in the app.
- Vehicle segmentation quality varies by photo. Inspect edges before finalising.
- Shadows are an adjustable approximate contact shadow, not physically simulated 3D lighting.
- The red 4:5 preview guide is for visual reference only. This version does not yet enforce the safe boundary.
- The uploaded showroom background is never regenerated; its dimensions remain the same.
- The vehicle cut-out is resized uniformly only, without turning or stretching it.
- Streamlit free hosting may be insufficient for larger image-segmentation workloads; a more powerful backend may be needed for production batches.
- For use with customer/vehicle photographs, choose hosting and account access appropriately.

## Planned next steps

Locked master background and all nine vehicle template references, template matching, crop safe placement, improved shadows, batch processing, better masking and secure staff access.

## Approved placement presets (Version 1.9)

The app has a **6 · Approved placement presets** section in the left sidebar.

1. Upload your Ford master background and vehicle, select the matching angle, and adjust placement and shadows until you approve the preview.
2. Click **Save current placement as approved**. This saves every relevant slider for that angle in the current Streamlit session; it does **not** save or publish photos.
3. Click **Download preset backup (.json)** and save the JSON file on your own computer. This is essential: Streamlit session state is temporary and can be cleared by a server restart or browser session change.
4. When returning later, upload that JSON under **Restore a preset backup (.json)**, click **Import preset backup**, select the matching angle, and click **Load approved placement for selected angle**.

A saved preset is an exact record of the current settings, not a guess reconstructed from screenshots. It is per angle; saving another angle does not overwrite a different angle. Import validates the supported numeric ranges. Approved layout is not automatically resized, and any crop-margin suggestions remain optional. Review the actual image before export.

Version 1.9 also adds a slightly more concentrated contact-shadow layer. Shadow quality is still approximate and depends on correctly placing tyre markers. Preset storage is currently limited to your session plus the downloadable JSON backup; permanent multi-user storage and login are future work.

## Full underbody shadows (Version 2.1)

The **4 · Floor shadow** panel now has:
- **Whole-vehicle underbody shadow (%)** — a separate soft chassis shadow spanning approximately the full width of the cut-out, not merely a line between two wheel markers
- **Underbody shadow depth (%)** — adjusts the projected depth of that broad floor shadow
- The original **Shadow strength**, **Shadow softness** and **Shadow width** controls still work
- **Add two more tyre contact shadows (optional)** — shows two extra adjustable tyre markers and tighter contact shadows for additional wheels, when visually appropriate

The underbody is a perspective-aware photographic approximation based on the known tyre contact positions. Review the floor under both axles and the chassis, because accurate photographic shadow direction requires user inspection. Every shadow is composited behind the uploaded original vehicle and over the fixed original showroom image, never into the vehicle pixels. The red roofline/crop guide and yellow marker labels remain preview-only. The program does not change the approved vehicle's scale, angle, or position when any shadow control is moved.

Previously downloaded Version 1.9 preset JSON files still import. New shadow controls receive defaults when importing old presets. After fine-tuning shadows, save the placement again and download a new JSON backup.

## Focused tyre-contact shadows (Version 2.3)

The underbody and cast-shadow engine stays the same, but each selected tyre now receives
**three distinct shadow components**: a broad, lightly blurred halo, a darker
medium-size patch and a small, sharply defined contact core exactly around the
wheel's estimated floor contact. The vehicle's colour, scale, crop and alignment
are untouched.

Under **4 · Floor shadow**, use the new **Tyre contact darkness (%)** slider
(default 150%) to strengthen wheel contacts independently of the full-body
shadow, and **Tyre contact footprint size (%)** (default 100%) to match the
shadow width to the tyre size. Under **Tyre contact positions (optional)**,
there are now independent **Tyre 1 contact fine left/right (pixels)** and
**Tyre 2 contact fine left/right (pixels)** controls. Use the existing vertical
offsets together with **Show tyre-shadow points on preview** to place each
marker where rubber really meets the tiles. These marker positions are only
visible in the preview; the clean 200-DPI PNG has no markers or guides.

Suggested starting values for the tested Ranger angle: general Shadow strength
100–110, softness 25–30 pixels, tyre contact darkness 145–165%, tyre contact
footprint size 90–105%. Verify both contacts before saving as approved; overly
high settings can produce unnatural dark spots.

Existing JSON placement backups from earlier versions remain importable:
all new wheel-shadow values receive safe defaults when missing. To keep the
new values, re-save the angle and download an updated preset JSON file.

## Four-wheel shadows for unseen tyres (Version 2.4)

**All four tyres contribute to the floor shadow, even when only two can be seen.**
The program estimates the two hidden-wheel contacts from the two visible wheel
contact points and a perspective offset for the chosen camera angle. Both
hidden-wheel patches are deliberately more subtle than visible-wheel patches,
and a four-corner floor quadrilateral helps connect them to the underbody shadow.
They do not create extra visible wheels on the vehicle itself.

Inside **Tyre contact positions (optional)**:
- **Include shadows from the two hidden tyres** defaults to enabled. Leave it on
  for normal four-wheel Ford photographs.
- **Hidden-wheel sideways perspective** adjusts whether the far-side wheels lie
  toward the left or right of the photograph. Default -14% is only a starting
  estimate for the tested front 30-degree view.
- **Hidden-wheel floor depth** adjusts how far up the showroom floor the far-side
  wheel contact points sit. Default 12% is a starting estimate.
- **Show tyre-shadow points on preview** shows yellow markers for the visible
  wheels and blue markers for inferred hidden contacts. This overlay is never
  included in exported images.
- **Manually override inferred hidden-wheel positions** is for unusual viewing
  angles only; otherwise leave it unticked to prevent incorrect dark spots
  around the bumper.

The inferred far-side wheel positions are not physically measured and will
need calibration per template angle. Existing placement presets are compatible
and default to four-wheel shadows when the new settings are absent. Your
approved vehicle crop, scale and roofline are unchanged by this update.

## Photo-reference shadow refinement (Version 2.5)

Based on visual comparison of the real grey Ranger showroom photo, the Ford
cropping template and the previous studio composite, the floor shadow now
behaves as a **continuous weight-bearing vehicle footprint** rather than four
separate dark circles.

- The two visible tyres retain concentrated, defined contact shadows.
- The two hidden tyres **still contribute**, but only by shaping a softly
  blurred quadrilateral between front/rear and near/far contact points. No
  independent high-contrast circles are drawn at the blue hidden-wheel markers.
- A front-weighted, softly blurred longitudinal mid-chassis band connects
  the engine/front bumper area, cabin and side step, tapering into the rear.
- The user-approved vehicle scale, roofline, crop, position and original
  showroom background remain unchanged.

New settings:
- **Front and mid-chassis shadow fullness (%)**, default 115. Increase
  moderately if too little shade appears along the middle or front underside.
- **Hidden-wheel shadow blend into chassis (%)**, default 55. Increase it for
  gentle far-side weight; decrease it if the underbody becomes too dark.
- Previously available visible tyre contact darkness/size and the inferred
  hidden-wheel perspective controls are unchanged.

Starting guidance for the tested front 30-degree Ranger: visible tyre contact
darkness around 150%, whole-vehicle underbody strength 100–115%, chassis
fullness 110–125%, hidden-wheel blend 45–65%. These are editorial starting
values rather than physically measured shadows. The app estimates geometry
from 2D images and requires final human visual review.

Old JSON presets still import: new fields receive defaults, and saving an
angle again plus downloading a new backup preserves the updated values.
Check that hidden-wheel support markers (blue) are roughly below the chassis,
even though no separate blue-marker patches should appear on the floor.

## Coherent hidden-wheel geometry and floor-only inspection (Version 2.6)

After visual review of the front 30-degree Ranger composite, the previous
hidden-wheel blue markers were misleading when drawn on top of the vehicle
body. A hidden tyre's contact location can be completely occluded by a door,
bumper or chassis in camera space even though its supporting ground shadow
belongs beneath the car.

**New behaviour**
- The two near-side visible tyre contacts remain anchored to the cut-out mask
  and can be adjusted independently as before.
- The far-side front tyre is inferred by a sideways-and-depth *axle* offset
  from the visible front tyre.
- The far-side rear tyre is derived from the far-side front contact using the
  observed near-side wheelbase vector, multiplied by **Far-side wheelbase
  perspective (%)** (default **95%**, range 70–110). This keeps both wheelbase
  vectors parallel, with a small amount of perspective foreshortening.
- The main composite shows only the **yellow markers for visible tyre contacts**,
  avoiding misleading blue markers drawn on top of body panels.
- With **Show tyre-shadow points on preview** enabled, expand **Inspect four-wheel
  floor footprint (vehicle hidden)** underneath the composite. This shows the
  full quadrilateral on the *blank showroom floor*: yellow visible tyre contacts,
  blue inferred hidden tyre supports, and connecting front/rear axle guide lines.
- The full four-wheel footprint still contributes to the connected underbody
  shading; dense local tyre shadows remain limited to the two visible tyres.
- Approved vehicle size, crop, roofline and placement do not move. Existing JSON
  presets still import; the new wheelbase setting defaults to 95%. Re-save the
  preset and download a new backup after calibrating the selected angle.

This remains an approximate 2.5D projection from a single vehicle photo, not
measured 3D wheel coordinates. Per-angle visual inspection against approved Ford
reference shots is still needed for accurate grounding.
