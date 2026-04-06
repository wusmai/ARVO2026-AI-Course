from typing import List, Dict
import matplotlib.pyplot as plt
from IPython.display import display  # Removed clear_output to stop flickering
import numpy as np
import torch
import time

def format_seconds(seconds: float) -> str:
    h, m, s = int(seconds) // 3600, (int(seconds) % 3600) // 60, int(seconds) % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"


def get_gpu_stats(device: torch.device) -> dict:
    stats = {"allocated_gb": np.nan, "reserved_gb": np.nan}
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or torch.cuda.current_device()
        stats["allocated_gb"] = torch.cuda.memory_allocated(idx) / (1024**3)
        stats["reserved_gb"] = torch.cuda.memory_reserved(idx) / (1024**3)
    return stats


class NotebookTrainingDashboard:
    #def __init__(self, out_path: str, train_smooth_window: int = 10):
    def __init__(self, out_path: str = "", train_smooth_window: int = 10):
        self.out_path = out_path
        self.train_smooth_window = train_smooth_window
        self.history = {
            k: []
            for k in [
                "train_x",
                "train_loss",
                "train_acc",
                "val_x",
                "val_loss",
                "val_acc",
                "val_auc",
                "lr_x",
                "lr",
                "gpu_x",
                "gpu_allocated_gb",
            ]
        }
        self.train_start_time = time.time()
        self.display_handle = display(display_id=True)

    @staticmethod
    def _smooth(values: List[float], window: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) < window:
            return arr
        return np.convolve(
            np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge"),
            np.ones(window) / window,
            mode="valid",
        )

    def update_train(self, x, train_loss, train_acc, lr, device, epoch_label, num_epochs, phase=""):
        gpu = get_gpu_stats(device)
        self.history["train_x"].append(x)
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["lr_x"].append(x)
        self.history["lr"].append(lr)
        self.history["gpu_x"].append(x)
        self.history["gpu_allocated_gb"].append(gpu["allocated_gb"])
        self._draw(epoch_label, num_epochs, device, phase)

    def update_val(self, epoch, train_loss, val_loss, train_acc, val_acc, val_auc, lr, device, num_epochs, phase=""):
        self.history["val_x"].append(float(epoch))
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["val_auc"].append(val_auc)
        self._draw(epoch, num_epochs, device, phase)

    def _draw(self, epoch_label, num_epochs, device, phase):
        elapsed = time.time() - self.train_start_time
        eta = (elapsed / max(epoch_label, 1)) * max(num_epochs - epoch_label, 0)
        gpu = get_gpu_stats(device)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        fig.suptitle(
            f"[{phase}] Epoch {epoch_label}/{num_epochs} | Elapsed: {format_seconds(elapsed)} | ETA: {format_seconds(eta)}\n"
            f"GPU Allocated: {gpu['allocated_gb']:.2f} GB",
            fontsize=12,
            fontweight="bold",
        )

        ax = axes[0, 0]
        if self.history["train_x"]:
            ax.plot(self.history["train_x"], self._smooth(self.history["train_loss"], self.train_smooth_window), label="Train")
        if self.history["val_x"]:
            ax.plot(self.history["val_x"], self.history["val_loss"], marker="o", label="Val")
        ax.set_title("Loss")
        ax.legend()

        ax = axes[0, 1]
        if self.history["train_x"]:
            ax.plot(self.history["train_x"], self._smooth(self.history["train_acc"], self.train_smooth_window), label="Train")
        if self.history["val_x"]:
            ax.plot(self.history["val_x"], self.history["val_acc"], marker="o", label="Val")
        ax.set_title("Accuracy")
        ax.legend()

        ax = axes[1, 0]
        if self.history["val_x"]:
            ax.plot(self.history["val_x"], self.history["val_auc"], marker="o", label="Val AUC", color="purple")
        ax.set_title("Validation AUC")

        ax = axes[1, 1]
        if self.history["lr_x"]:
            ax.plot(self.history["lr_x"], self.history["lr"], label="LR", color="orange")
        ax.set_title("Learning Rate")

        plt.tight_layout()

        if self.display_handle:
            self.display_handle.update(fig)

        #fig.savefig(self.out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

