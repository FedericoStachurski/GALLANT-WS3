import io
import os
import json
from typing import Optional, Sequence, Dict, Any

import requests
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


def build_caption_from_row(row) -> str:
    # Prefer description if present, else fallback to title
    desc = str(row.get("description", "")).strip()
    title = desc if desc else str(row.get("title", "")).strip()

    # Your original tag logic unchanged
    tags = str(row.get("tags", "")).strip() if "tags" in row else ""
    tags_list = [t for t in tags.split() if t.lower() != "glasgow"]
    extra = " ".join(tags_list[:3])

    # Same return logic as before
    if extra:
        return f"{title}. {extra}".strip()

    return title



class ImageCaptioningDataset(Dataset):
    def __init__(self, df, url_col: str = "url_sq", caption_builder=build_caption_from_row):
        self.df = df.reset_index(drop=True)
        self.url_col = url_col
        self.caption_builder = caption_builder

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _has_valid_url(url) -> bool:
        if url is None:
            return False
        if isinstance(url, float):
            return False
        url = str(url).strip()
        if not url or url.lower() == "nan":
            return False
        return True

    @staticmethod
    def _load_image(url: str) -> Image.Image:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {url}: {e}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = row[self.url_col]

        if not self._has_valid_url(url):
            return None

        try:
            image = self._load_image(url)
        except:
            return None

        text = self.caption_builder(row)
        return {"image": image, "text": text}



class VisionLanguageAdapterTrainer:
    """
    Vision–language LoRA trainer (BLIP for now) with saving of:
      - adapter weights
      - training loss history
      - metadata/config

    Everything is saved under: <output_root>/<run_name>/
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        model_family: str = "blip",
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[Sequence[str]] = None,
        lr: float = 5e-5,
        num_epochs: int = 3,
        batch_size: int = 16,
        device: Optional[str] = None,
        output_root: str = "models",
        run_name: str = "blip_run",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model_name: HF model ID.
            model_family: "blip" (for now).
            lora_r, lora_alpha, lora_dropout: LoRA hyperparams.
            lora_target_modules: list of substrings of module names to target.
            lr: learning rate.
            num_epochs: training epochs.
            batch_size: default batch size for create_dataloader().
            device: "cuda" / "cpu" / None (auto).
            output_root: parent folder in which to save models ("models").
            run_name: subfolder name for this specific training run.
            extra_metadata: optional dict to inject custom info into metadata.json.
        """
        self.model_name = model_name
        self.model_family = model_family.lower()
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.output_root = output_root
        self.run_name = run_name
        self.extra_metadata = extra_metadata or {}

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[INFO] Using device: {self.device}")

        # Load model & processor
        self.processor = self._load_processor()
        self.model = self._load_model()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        # LoRA config
        if lora_target_modules is None:
            lora_target_modules = [
                "self.query",
                "self.key",
                "self.value",
            ]

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=list(lora_target_modules),
        )

        self._apply_lora()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )

        self.losses = []  # average loss per epoch

    # ---------------- Model loading ---------------- #

    def _load_processor(self):
        if self.model_family == "blip":
            return AutoProcessor.from_pretrained(self.model_name)
        raise NotImplementedError(f"Model family '{self.model_family}' not implemented yet.")

    def _load_model(self) -> nn.Module:
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.model_family == "blip":
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            ).to(self.device)
        else:
            raise NotImplementedError(f"Model family '{self.model_family}' not implemented yet.")

        # freeze base
        for p in model.parameters():
            p.requires_grad = False

        return model

    def _apply_lora(self):
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    # ---------------- DataLoader & collator ---------------- #

    def create_dataloader(
        self,
        df,
        url_col: str = "url_sq",
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        dataset = ImageCaptioningDataset(df, url_col=url_col)

        if batch_size is None:
            batch_size = self.batch_size

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collator,
        )
        return loader

    def _collator(self, batch):
        # Remove any invalid entries
        batch = [item for item in batch if item is not None]

        if len(batch) == 0:
            return None  # let training skip the batch

        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]

        try:
            image_inputs = self.processor(
                images=images,
                padding="max_length",
                return_tensors="pt",
            )

            text_inputs = self.processor.tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
            )

            return {
                "pixel_values": image_inputs["pixel_values"],
                "input_ids": text_inputs["input_ids"],
                "attention_mask": text_inputs["attention_mask"],
            }

        except Exception as e:
            # If anything goes wrong, skip this batch
            print(f"[WARN] Collator skipping batch due to: {e}")
            return None


    # ---------------- Training ---------------- #

    def train(self, train_loader: DataLoader):
        self.model.train()
        epoch_losses = []

        for epoch in range(self.num_epochs):
            temp_loss = []
            print(f"\n=== Epoch {epoch} ===")

            for step, batch in tqdm(
                enumerate(train_loader),
                desc="Training steps",
                leave=False,
            ):
                if batch is None:
                    continue

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids,
                    attention_mask=attention_mask,
                )

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                temp_loss.append(loss.item())

            avg_loss = sum(temp_loss) / max(len(temp_loss), 1)
            epoch_losses.append(avg_loss)
            print(f"  epoch {epoch} || Average training loss: {avg_loss:.4f}")

        self.losses = epoch_losses
        return epoch_losses

    # ---------------- Saving ---------------- #

    def _build_save_dir(self) -> str:
        """
        Returns the full path: <output_root>/<run_name>/
        """
        save_dir = os.path.join(self.output_root, self.run_name)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    
    @staticmethod
    def _to_jsonable(obj):
        """
        Recursively convert objects to JSON-serializable types.
        - sets -> lists
        - tuples -> lists
        - objects with __dict__ -> dict
        - everything else -> str as a last resort
        """
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        if isinstance(obj, dict):
            return {str(k): VisionLanguageAdapterTrainer._to_jsonable(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [VisionLanguageAdapterTrainer._to_jsonable(v) for v in obj]

        if hasattr(obj, "__dict__"):
            return VisionLanguageAdapterTrainer._to_jsonable(obj.__dict__)

        # fallback: string representation
        return str(obj)

    def save(self):
        """
        Save:
          - adapter weights + processor: <save_dir>/adapter/
          - training losses: <save_dir>/training_losses.json
          - metadata/config: <save_dir>/metadata.json
        """
        save_dir = self._build_save_dir()
        adapter_dir = os.path.join(save_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)

        # 1) Save the PEFT adapter + processor
        # This saves only the adapter weights (and PEFT config), not the original BLIP.
        self.model.save_pretrained(adapter_dir)
        self.processor.save_pretrained(adapter_dir)

        # 2) Save training loss history
        losses_path = os.path.join(save_dir, "training_losses.json")
        with open(losses_path, "w") as f:
            json.dump(self.losses, f, indent=2)

        # 3) Save metadata/config
        meta = {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "device": self.device,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "lora_config": self.lora_config.to_dict(),
            "final_loss": self.losses[-1] if self.losses else None,
            "output_root": self.output_root,
            "run_name": self.run_name,
        }

        
        meta.update(self.extra_metadata)

        def make_json_safe(obj):
            """
            Recursively convert objects to JSON-serializable types:
            - set -> list
            - tuple -> list
            - dict/list -> walk recursively
            """
            # basic JSON types
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj

            # dict: fix values (and keys just in case)
            if isinstance(obj, dict):
                return {str(k): make_json_safe(v) for k, v in obj.items()}

            # list/tuple/set -> list
            if isinstance(obj, (list, tuple, set)):
                return [make_json_safe(v) for v in obj]

            # objects with __dict__ (e.g. configs) -> dict
            if hasattr(obj, "__dict__"):
                return make_json_safe(vars(obj))

            # fallback: string representation
            return str(obj)

        meta_safe = make_json_safe(meta)

        meta_path = os.path.join(save_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta_safe , f, indent=2)

        print(f"[INFO] Saved adapter + metadata to: {save_dir}")

    # ---------------- Inference helper ---------------- #

    @staticmethod
    def _load_image_from_url(url: str) -> Image.Image:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def generate_caption(self, url: str, prompt: str = " ") -> str:
        self.model.eval()

        image = self._load_image_from_url(url)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
