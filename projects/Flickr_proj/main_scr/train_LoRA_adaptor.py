import os
import argparse
import pandas as pd  # <-- NEW

from projects.utilities_scr.load_data_flickr import FlickrDataLoader
from projects.Flickr_proj.build_caption_model_adaptor import (
    VisionLanguageAdapterTrainer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BLIP LoRA adapter on Flickr West End data"
    )

    # Data-related
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/staff3/fstachurski/Flickr_data",
        help="Folder containing west_end_*_df_flickr.xlsx files",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default="west_end_",
        help="Substring that filenames must contain",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        default="_df_flickr.xlsx",
        help="Suffix of the Excel files",
    )

    parser.add_argument(
        "--url-priority",
        type=str,
        default="url_sq",
        help="Which Flickr URL column name to use inside the DataFrame "
             "(e.g. url_sq, url_m, url_l). We'll map image_url → this name.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="If >0, random sample that many items. If 0, use ALL.",
    )

    # Training-related
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--run-name",
        type=str,
        default="blip_westend_full",
        help="Name used for the model folder inside models/",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="models",
        help="Root directory where model runs are stored",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---------- 1. Load Flickr items ----------
    print("[INFO] Loading Flickr items for training...")

    loader = FlickrDataLoader(
        data_dir=args.data_dir,
        name_contains=args.name_contains,
        file_suffix=args.file_suffix,
        url_priority=[args.url_priority],  # e.g. ["url_sq"]
        sample_size=None if args.sample_size == 0 else args.sample_size,
        check_urls=False,  # skip HTTP pinging for speed during training
    )

    items = loader.build_items()
    print(f"[INFO] Total items loaded: {len(items)}")

    if len(items) == 0:
        raise RuntimeError("No items loaded — check data_dir / filters.")

    # ---------- 2. Convert items -> DataFrame for the trainer ----------
    # items should look like: {"image_url": ..., "caption": ..., ...}
    df = pd.DataFrame(items)

    if "image_url" not in df.columns:
        raise RuntimeError("Expected 'image_url' key in items but it is missing.")

    # Map "image_url" -> the column name the trainer expects (e.g. "url_sq")
    url_col_name = args.url_priority
    df[url_col_name] = df["image_url"]

    # The trainer's ImageCaptioningDataset builds text via build_caption_from_row,
    # which uses row['title'] and row['tags']. If they don't exist, we create them.
    if "title" not in df.columns:
        # Use the existing caption as title if present, otherwise empty string
        df["title"] = df.get("caption", "")

    if "tags" not in df.columns:
        df["tags"] = ""

    # Optional: filter to valid URLs only, if your items have e.g. ok_image flag
    if "ok_image" in df.columns:
        before = len(df)
        df = df[df["ok_image"] == True].reset_index(drop=True)
        print(f"[INFO] Filtered by ok_image: {before} -> {len(df)} rows")

    print(f"[INFO] Final training DataFrame size: {len(df)}")

    # ---------- 3. Set up trainer ----------
    trainer = VisionLanguageAdapterTrainer(
        model_name="Salesforce/blip-image-captioning-base",
        model_family="blip",
        device=None,  # let class pick CUDA if available
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name=args.run_name,
        output_root=args.output_root,
        extra_metadata={
            "data_dir": args.data_dir,
            "name_contains": args.name_contains,
            "file_suffix": args.file_suffix,
            "url_priority": args.url_priority,
            "num_items": len(df),
        },
    )

    # ---------- 4. Build DataLoader from DataFrame ----------
    train_loader = trainer.create_dataloader(
        df,
        url_col=url_col_name,        # e.g. "url_sq"
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,               # set >0 on SLURM if you want
    )

    # ---------- 5. Train ----------
    print("[INFO] Starting training...")
    losses = trainer.train(train_loader)
    print("[INFO] Training losses per epoch:", losses)

    # ---------- 6. Save adapter + metadata ----------
    print("[INFO] Saving adapter and metadata...")
    trainer.save()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

