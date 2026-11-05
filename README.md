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
