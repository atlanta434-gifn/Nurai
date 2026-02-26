import streamlit as st
from pathlib import Path
import json
import zipfile
import io

from quantly.vision.synthetic import generate_shapes_dataset
from quantly.vision.augmentation import AugmentConfig
from quantly.vision.dataset import build_augmented_dataset
from quantly.vision.train_classifier import TrainConfig, train_classifier

st.set_page_config(page_title="Quantly Vision", page_icon="🖼️", layout="wide")

ROOT = Path(".")
IMAGES_ROOT = ROOT / "data" / "images"
OUTPUT_ROOT = ROOT / "data" / "output"
MODELS_ROOT = ROOT / "models"

st.title("🖼️ Quantly: Image Dataset Generator + Vision Training (Streamlit)")

with st.sidebar:
    st.header("Vision Settings")
    copies_per_image = st.slider("Augmentation copies per image", 0, 10, 2)
    val_ratio = st.slider("Validation split ratio", 0.05, 0.5, 0.2, 0.05)
    st.markdown("---")
    st.subheader("Training")
    epochs = st.slider("Epochs", 1, 20, 3)
    batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=2)
    image_size = st.selectbox("Image size", [64, 96, 128, 160, 224], index=2)
    lr = st.selectbox("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], index=2)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)

tab1, tab2, tab3 = st.tabs(["1) Add Images", "2) Generate & Export Dataset", "3) Train Vision Model"])

with tab1:
    st.subheader("Upload images into class folders")
    st.write("Your images should be organized like: `data/images/<class_name>/*.png`")
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    colA, colB = st.columns(2)
    with colA:
        new_class = st.text_input("Create class folder")
        if st.button("Create class"):
            if new_class.strip():
                (IMAGES_ROOT / new_class.strip()).mkdir(parents=True, exist_ok=True)
                st.success(f"Created: {new_class.strip()}")
            else:
                st.warning("Enter a class name.")

    with colB:
        st.caption("Quick demo dataset (no internet needed)")
        n_per_class = st.number_input("Synthetic images per class", 5, 300, 40, 5)
        if st.button("Generate synthetic demo dataset (shapes)"):
            created = generate_shapes_dataset(IMAGES_ROOT, n_per_class=int(n_per_class), image_size=128, seed=1)
            st.success(f"Generated {len(created)} images into {IMAGES_ROOT}")

    st.markdown("---")
    classes = sorted([p.name for p in IMAGES_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not classes:
        st.info("No class folders yet. Create one or generate the demo dataset.")
    else:
        selected_class = st.selectbox("Select class to upload into", classes)
        files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        if files and selected_class:
            out_dir = IMAGES_ROOT / selected_class
            for f in files:
                out_path = out_dir / Path(f.name).name
                out_path.write_bytes(f.getvalue())
            st.success(f"Saved {len(files)} images to {out_dir}")

        # show counts
        st.write("Current dataset:")
        for c in classes:
            count = len(list((IMAGES_ROOT/c).glob("*.*")))
            st.write(f"- **{c}**: {count} image(s)")

with tab2:
    st.subheader("Generate augmented dataset + export")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    cfg = AugmentConfig(
        flips=True, rotate=True, brightness=True, contrast=True,
        max_rotate_deg=25, seed=42
    )

    if st.button("Build augmented dataset + train/val split"):
        out = build_augmented_dataset(
            images_root=IMAGES_ROOT,
            output_root=OUTPUT_ROOT / "vision_dataset",
            copies_per_image=int(copies_per_image),
            aug_cfg=cfg,
            val_ratio=float(val_ratio),
            seed=42
        )
        st.session_state["last_build"] = {k: str(v) for k, v in out.items()}
        st.success("Dataset built successfully.")
        st.json(st.session_state["last_build"])

    if "last_build" in st.session_state:
        st.markdown("### Export as ZIP")
        build_paths = {k: Path(v) for k, v in st.session_state["last_build"].items()}
        dataset_root = build_paths["output_root"]

        def zip_dir_to_bytes(root: Path) -> bytes:
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in root.rglob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(root))
            return mem.getvalue()

        zip_bytes = zip_dir_to_bytes(dataset_root)
        st.download_button(
            "Download dataset ZIP",
            data=zip_bytes,
            file_name="vision_dataset.zip",
            mime="application/zip"
        )

        st.markdown("### Preview (first 20 lines)")
        try:
            train_csv = (dataset_root / "splits" / "train.csv").read_text(encoding="utf-8").splitlines()[:20]
            st.code("\n".join(train_csv))
        except Exception as e:
            st.warning(f"Could not preview CSV: {e}")

with tab3:
    st.subheader("Train a small CNN classifier (PyTorch)")
    dataset_root = OUTPUT_ROOT / "vision_dataset"
    train_dir = dataset_root / "splits" / "train"
    val_dir = dataset_root / "splits" / "val"

    if not train_dir.exists() or not val_dir.exists():
        st.info("Build the dataset first in tab 2.")
    else:
        model_name = st.text_input("Model output name", "vision_classifier.pt")
        if st.button("Train model"):
            cfg = TrainConfig(
                image_size=int(image_size),
                batch_size=int(batch_size),
                epochs=int(epochs),
                lr=float(lr),
                device=str(device),
            )
            out_path = MODELS_ROOT / model_name
            with st.spinner("Training..."):
                metrics = train_classifier(train_dir, val_dir, out_path, cfg)
            st.success(f"Saved model to: {out_path}")
            st.json(metrics)

        # show existing models
        MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        models = sorted([p.name for p in MODELS_ROOT.glob("*.pt")])
        if models:
            st.markdown("### Existing models")
            st.write(models)

st.markdown("---")
st.caption("Deploy: push to GitHub, then Streamlit Cloud -> select streamlit_app.py")
