"""
flickr_utils.py

Utilities for loading and cleaning Flickr-based image–caption data,
plus a simple PyTorch Dataset for captioning models.
"""

import argparse
import glob
import io
from typing import List, Dict, Optional, Sequence

import pandas as pd
import requests
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------
# Flickr URL checking
# ---------------------------------------------------------------------

def check_flickr_url(url: str) -> Dict:
    """
    Check whether a Flickr static URL returns a real image.

    Returns a dict with keys:
        - status: HTTP status code or None
        - ok_image: bool, True if looks like a valid image
        - droids_page: bool, True if it's the Flickr 'droids' error page
        - content_type: str, response Content-Type
        - message / error: optional strings
    """
    # Handle missing or null URLs directly
    if not url or not isinstance(url, str):
        return {
            "status": None,
            "ok_image": False,
            "droids_page": False,
            "content_type": "",
            "message": "No URL provided",
        }

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; flickr-url-checker/1.0)"
    }

    try:
        resp = requests.get(url, timeout=8, headers=headers)

        # If not 200, it's invalid
        if resp.status_code != 200:
            return {
                "status": resp.status_code,
                "ok_image": False,
                "droids_page": False,
                "content_type": resp.headers.get("Content-Type", ""),
                "message": "Non-200 status",
            }

        content_type = resp.headers.get("Content-Type", "").lower()

        # Flickr 'droids' page = HTML error page text
        droids = b"These aren't the droids you're looking for" in resp.content

        # A valid image must:
        # - not be a droids page
        # - content type contains "image" OR "octet-stream"
        is_image = (("image" in content_type or "octet-stream" in content_type)
                    and not droids)

        return {
            "status": resp.status_code,
            "ok_image": is_image,
            "droids_page": droids,
            "content_type": content_type,
        }

    except Exception as e:
        return {
            "status": None,
            "ok_image": False,
            "droids_page": False,
            "content_type": "",
            "error": str(e),
        }


# ---------------------------------------------------------------------
# Flickr data loader class
# ---------------------------------------------------------------------

class FlickrDataLoader:
    """
    A helper for loading Flickr Excel exports, building captions,
    choosing a URL column, and (optionally) checking URLs.

    Typical usage:

        loader = FlickrDataLoader(
            data_dir="/home/.../Flickr_data",
            name_contains="west_end_",
            file_suffix="_df_flickr.xlsx",
            url_priority=["url_sq"],
            sample_size=100,
            check_urls=True,
        )
        items = loader.build_items()
    """

    def __init__(
        self,
        data_dir: str,
        name_contains: Optional[str] = None,
        file_suffix: str = "_df_flickr.xlsx",
        url_priority: Optional[Sequence[str]] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42,
        check_urls: bool = False,
    ):
        """
        Args:
            data_dir: Folder where the Excel files live.
            name_contains: Substring that must appear in the filename
                           (e.g. "west_end_"). If None, all matching files are used.
            file_suffix: Pattern at the end of each file name
                         (e.g. "_df_flickr.xlsx").
            url_priority: Ordered list of URL columns to use, e.g.
                          ["url_sq", "url_l", "url_c"]. The first present & valid
                          one in each row is chosen. If None, defaults to:
                          ["url_sq", "url_l", "url_c", "url_z", "url_m", "url_n", "url_o"].
            sample_size: If set, random subsample of rows to keep from the
                         concatenated DataFrame.
            random_state: Random state for sampling.
            check_urls: If True, run HTTP checks to verify that URLs are
                        valid images and attach 'ok_image' flags.
        """
        self.data_dir = data_dir
        self.name_contains = name_contains
        self.file_suffix = file_suffix

        if url_priority is None:
            self.url_priority = [
                "url_sq",
                "url_l",
                "url_c",
                "url_z",
                "url_m",
                "url_n",
                "url_o",
            ]
        else:
            self.url_priority = list(url_priority)

        self.sample_size = sample_size
        self.random_state = random_state
        self.check_urls = check_urls

    # -------------------------
    # File discovery & loading
    # -------------------------

    def find_files(self) -> List[str]:
        """
        Find all Excel files in data_dir that match the specified pattern.
        """
        # Start with all xlsx in the directory
        pattern = f"{self.data_dir}/*.xlsx"
        files = glob.glob(pattern)

        # Filter by substring if requested
        if self.name_contains:
            files = [f for f in files if self.name_contains in f]

        # Filter by suffix if requested
        if self.file_suffix:
            files = [f for f in files if f.endswith(self.file_suffix)]

        return sorted(files)

    def load_dataframe(self) -> pd.DataFrame:
        """
        Load and concatenate all matching Excel files into a single DataFrame.
        Applies optional subsampling if sample_size is set.
        """
        files = self.find_files()
        if not files:
            raise FileNotFoundError(
                f"No Excel files found in {self.data_dir} with "
                f"name_contains={self.name_contains!r}, file_suffix={self.file_suffix!r}"
            )

        dfs = [pd.read_excel(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        if self.sample_size is not None and self.sample_size < len(df):
            df = df.sample(self.sample_size, random_state=self.random_state).reset_index(drop=True)

        return df

    # -------------------------
    # Caption building
    # -------------------------

    @staticmethod
    def build_caption(title, tags) -> str:
        """
        Build a simple caption from title and tags.

        - Casts to string and strips whitespace.
        - Removes the tag 'glasgow' (case-insensitive).
        - Keeps up to 3 other tags.
        - Returns: "title. tag1 tag2 tag3"
        """
        title = str(title).strip()
        tags = str(tags).strip()

        tag_list = [t for t in tags.split() if t.lower() != "glasgow"]
        extra = " ".join(tag_list[:3])

        base = f"{title}. {extra}".strip()
        # Remove trailing '.' if we ended up with just the title.
        return base.strip(". ").strip()

    # -------------------------
    # URL selection
    # -------------------------

    def row_to_item(self, row: pd.Series, df_columns) -> Optional[Dict]:
        """
        Convert a DataFrame row into an item dict with 'image_url' and 'caption'.

        Returns None if no usable URL is found.
        """
        # Choose URL according to priority
        url = None
        for col in self.url_priority:
            if col in df_columns:
                val = row.get(col)
                if isinstance(val, str) and val.startswith("http"):
                    url = val
                    break

        if not url:
            return None

        # Build caption
        title = row.get("title", "")
        tags = row.get("tags", "")
        caption = self.build_caption(title, tags)

        return {
            "image_url": url,
            "caption": caption,
        }

    def df_to_items(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert a DataFrame into a list of item dicts:
            {"image_url": ..., "caption": ...}
        Skips rows with no usable URL.
        """
        items: List[Dict] = []
        cols = df.columns

        for _, row in df.iterrows():
            item = self.row_to_item(row, cols)
            if item is not None:
                items.append(item)

        return items

    # -------------------------
    # URL filtering & building
    # -------------------------

    @staticmethod
    def clean_items(items: List[Dict]) -> List[Dict]:
        """
        Remove items with missing / invalid image_url strings
        (None, '', 'nan').
        """
        clean = []
        for it in items:
            url = it.get("image_url")
            if url is None:
                continue
            url_str = str(url).strip()
            if not url_str:
                continue
            if url_str.lower() == "nan":
                continue
            # keep item
            clean.append(it)
        return clean

    def attach_url_checks(self, items: List[Dict]) -> List[Dict]:
        """
        Run HTTP checks on each URL and attach 'ok_image' etc.
        """
        results = []
        for item in items:
            res = check_flickr_url(item["image_url"])
            merged = {**item, **res}
            results.append(merged)
        return results

    def filter_ok_images(self, items: List[Dict]) -> List[Dict]:
        """
        Keep only items marked as ok_image=True.
        If that key is missing (e.g., check_urls=False), all items are kept.
        """
        filtered = []
        for it in items:
            ok_flag = it.get("ok_image")
            if ok_flag is None:
                # if we never ran checks, keep it
                filtered.append(it)
            elif ok_flag:
                filtered.append(it)
        return filtered

    # -------------------------
    # Full pipeline
    # -------------------------

    def build_items(self) -> List[Dict]:
        """
        Full pipeline:
          1) Load DataFrame from Excel(s)
          2) Convert rows to items with URL + caption
          3) Remove obviously bad URLs (None / '' / 'nan')
          4) (Optional) check URLs via HTTP and drop invalid ones

        Returns:
            List[dict] where each dict has at least:
                - 'image_url'
                - 'caption'
              and, if check_urls=True, also:
                - 'status', 'ok_image', 'droids_page', 'content_type'
        """
        df = self.load_dataframe()
        items = self.df_to_items(df)
        items = self.clean_items(items)

        if self.check_urls:
            items = self.attach_url_checks(items)
            items = self.filter_ok_images(items)

        return items


# ---------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------

class CaptionDataset(Dataset):
    """
    Simple PyTorch dataset that takes pre-built items (image_url + caption)
    and a HuggingFace-style processor to prepare model inputs.

    Example:
        dataset = CaptionDataset(items, processor)
        batch = dataset[0]
    """

    def __init__(self, items, processor, max_length: int = 32):
        self.items = items
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _load_image(url: str) -> Image.Image:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def __getitem__(self, idx):
        item = self.items[idx]
        image = self._load_image(item["image_url"])
        caption = item["caption"]

        inputs = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Standard seq2seq-style labels = input_ids shifted by model
        inputs["labels"] = inputs["input_ids"].clone()
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs


# ---------------------------------------------------------------------
# CLI entrypoint for quick testing
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test FlickrDataLoader on a folder of Excel files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Folder containing Flickr Excel files.",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default=None,
        help="Substring that must appear in the filename (e.g. 'west_end_').",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        default="_df_flickr.xlsx",
        help="Filename suffix to match (default: '_df_flickr.xlsx').",
    )
    parser.add_argument(
        "--url-priority",
        type=str,
        default="url_sq",
        help=(
            "Comma-separated list of URL columns in priority order, "
            "e.g. 'url_sq,url_l,url_c'. Default: 'url_sq'."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of rows to sample from the combined DataFrame.",
    )
    parser.add_argument(
        "--check-urls",
        action="store_true",
        help="If set, perform HTTP checks on URLs and drop invalid images.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Number of example items to print.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    url_priority = [c.strip() for c in args.url_priority.split(",") if c.strip()]

    loader = FlickrDataLoader(
        data_dir=args.data_dir,
        name_contains=args.name_contains,
        file_suffix=args.file_suffix,
        url_priority=url_priority,
        sample_size=args.sample_size,
        check_urls=args.check_urls,
    )

    print("\n[INFO] Discovering files...")
    files = loader.find_files()
    print(f"  Found {len(files)} file(s):")
    for f in files:
        print("   -", f)

    print("\n[INFO] Building items...")
    items = loader.build_items()
    print(f"[INFO] Built {len(items)} item(s).")

    # Show a few examples
    n_show = min(args.show_examples, len(items))
    for i in range(n_show):
        it = items[i]
        print(f"\n=== Example {i} ===")
        print("URL:    ", it["image_url"])
        print("Caption:", it["caption"])
        if "ok_image" in it:
            print("ok_image:", it["ok_image"], "| status:", it.get("status"))
