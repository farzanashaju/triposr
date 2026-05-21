# python batch.py \
#     --input-dir /home2/farzana/3d-heritage/data \
#     --output-dir /home2/farzana/3d-heritage/output_triposr --mc-resolution 256

import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import trimesh

from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation


class Timer:
    def __init__(self):
        self.items = {}

    def start(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - self.items.pop(name)
        logging.info(f"{name} finished in {elapsed:.2f}s")


timer = Timer()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-dir",
    required=True,
    type=str,
)

parser.add_argument(
    "--output-dir",
    required=True,
    type=str,
)

parser.add_argument(
    "--device",
    default="cuda:0",
)

parser.add_argument(
    "--pretrained-model-name-or-path",
    default="stabilityai/TripoSR",
)

parser.add_argument(
    "--chunk-size",
    type=int,
    default=8192,
)

parser.add_argument(
    "--mc-resolution",
    type=int,
    default=256,
)

parser.add_argument(
    "--foreground-ratio",
    type=float,
    default=0.85,
)

parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will not be removed."
)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

timer.start("Initializing model")

model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)

model.renderer.set_chunk_size(args.chunk_size)
model.to(device)

timer.end("Initializing model")

valid_exts = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

image_paths = sorted(
    [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if os.path.splitext(f.lower())[1] in valid_exts
    ]
)

logging.info(f"Found {len(image_paths)} images")

rembg_session = rembg.new_session()

for image_path in image_paths:

    stem = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    logging.info(f"Processing {stem}")

    image = Image.open(image_path)

    if not args.no_remove_bg:
        image = image.convert("RGB")
        image = remove_background(
            image,
            rembg_session,
        )
        image = resize_foreground(
            image,
            args.foreground_ratio,
        )

    if image.mode == "RGBA" or (isinstance(image, np.ndarray) and image.shape[-1] == 4) or getattr(image, 'mode', '') == 'RGBA':
        image = np.array(image).astype(np.float32) / 255.0
        image = (
            image[:, :, :3] * image[:, :, 3:4]
            + (1 - image[:, :, 3:4]) * 0.5
        )
        image = Image.fromarray(
            (image * 255.0).astype(np.uint8)
        )
    else:
        image = image.convert("RGB")

    timer.start(f"{stem}: inference")

    with torch.no_grad():
        scene_codes = model(
            [image],
            device=device,
        )

    timer.end(f"{stem}: inference")

    timer.start(f"{stem}: mesh extraction")

    meshes = model.extract_mesh(
        scene_codes,
        has_vertex_color=True,
        resolution=args.mc_resolution,
    )

    mesh = meshes[0]
    mesh = to_gradio_3d_orientation(mesh)

    timer.end(f"{stem}: mesh extraction")

    glb_path = os.path.join(
        args.output_dir,
        f"{stem}.glb",
    )

    mesh.export(glb_path)

    logging.info(
        f"Saved:\n"
        f"  {glb_path}"
    )

logging.info("Done.")