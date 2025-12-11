import numpy as np
from PIL import Image
from pathlib import Path

def extract_pv_masks(
    src_dir: Path,
    dst_dir: Path,
    pv_class: int = 0,
    background_value: int = 255
):
    """
    Extract masks that contain PV modules (class 0) and save a cleaned PV-only mask.

    Parameters
    ----------
    src_dir : Path
        Directory containing original RID superstructure masks.
    dst_dir : Path
        Output directory for pv-only masks.
    pv_class : int, optional
        Class index representing PV modules (default: 0).
    background_value : int, optional
        Value used for non-PV pixels in the output mask (default: 255 for easy visualization).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    mask_files = list(src_dir.glob("*.png"))

    print(f"Found {len(mask_files)} masks to inspect...")
    count_saved = 0

    for mask_path in mask_files:
        mask = np.array(Image.open(mask_path))

        # Check whether PV class exists
        if pv_class not in np.unique(mask):
            continue  # skip masks without PV

        # Create PV-only mask
        pv_mask = np.where(mask == pv_class, pv_class, background_value).astype(np.uint8)

        # Save to new directory
        out_path = dst_dir / mask_path.name
        Image.fromarray(pv_mask).save(out_path)
        count_saved += 1

    print(f"Saved {count_saved} PV-only masks to: {dst_dir}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    SRC_MASK_DIR = BASE_DIR / "masks_superstructures_reviewed"
    DST_PV_MASK_DIR = BASE_DIR / "masks_pv_modules_only"

    extract_pv_masks(
        src_dir=SRC_MASK_DIR,
        dst_dir=DST_PV_MASK_DIR,
        pv_class=0,
        background_value=255,   # white background for visualization
    )