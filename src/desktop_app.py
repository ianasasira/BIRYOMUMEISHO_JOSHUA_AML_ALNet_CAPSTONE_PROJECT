import os
import sys
from pathlib import Path

import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_logger import init_db, log_prediction, get_recent_logs, get_stats
from alnet_model import ALNet


def _resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent.parent
    return base / relative_path


MODEL_DIR = _resource_path("outputs")
IMG_SIZE = 224
CLASS_NAMES = ["Non-AML", "AML"]
AML_THRESHOLD = 0.15

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class ALNetDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ALNet — AML Screening Tool")
        self.geometry("1000x700")
        self.minsize(900, 600)

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.tab_analyze = self.tabview.add("Analyze")
        self.tab_history = self.tabview.add("History")

        self._build_analyze_tab()
        self._build_history_tab()

    def _build_analyze_tab(self):
        self.tab_analyze.grid_columnconfigure(0, weight=1)
        self.tab_analyze.grid_columnconfigure(1, weight=1)
        self.tab_analyze.grid_rowconfigure(1, weight=1)

        header = ctk.CTkLabel(
            self.tab_analyze, text="ALNet AML Detection",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        header.grid(row=0, column=0, columnspan=2, pady=(10, 5), sticky="n")

        subtitle = ctk.CTkLabel(
            self.tab_analyze, text="Decision-Support Screening Tool — Not a Diagnosis",
            font=ctk.CTkFont(size=13), text_color="gray"
        )
        subtitle.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky="n")

        # Left panel: image
        left_frame = ctk.CTkFrame(self.tab_analyze)
        left_frame.grid(row=2, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            left_frame, text="Blood Smear Image",
            font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, pady=(10, 5))

        self.image_preview = ctk.CTkLabel(left_frame, text="", width=350, height=350)
        self.image_preview.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.btn_load = ctk.CTkButton(
            left_frame, text="Load Image", command=self._load_image,
            width=200, height=40, font=ctk.CTkFont(size=14)
        )
        self.btn_load.grid(row=2, column=0, pady=(5, 15))

        # Right panel: results
        right_frame = ctk.CTkFrame(self.tab_analyze)
        right_frame.grid(row=2, column=1, sticky="nsew", padx=(5, 10), pady=(0, 10))
        right_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right_frame, text="Analysis Result",
            font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, pady=(10, 10))

        self.result_prediction = ctk.CTkLabel(
            right_frame, text="—",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.result_prediction.grid(row=1, column=0, pady=(5, 5))

        self.result_confidence = ctk.CTkLabel(
            right_frame, text="",
            font=ctk.CTkFont(size=16)
        )
        self.result_confidence.grid(row=2, column=0, pady=(0, 5))

        self.progress_bar = ctk.CTkProgressBar(right_frame, width=280)
        self.progress_bar.grid(row=3, column=0, pady=(5, 15))
        self.progress_bar.set(0)

        self.confidence_labels_frame = ctk.CTkFrame(right_frame)
        self.confidence_labels_frame.grid(row=4, column=0, pady=(5, 10), sticky="ew")
        self.confidence_labels_frame.grid_columnconfigure((0, 1), weight=1)

        self.lbl_aml_conf = ctk.CTkLabel(
            self.confidence_labels_frame, text="AML: —",
            font=ctk.CTkFont(size=15)
        )
        self.lbl_aml_conf.grid(row=0, column=0, padx=10, pady=5)

        self.lbl_non_aml_conf = ctk.CTkLabel(
            self.confidence_labels_frame, text="Non-AML: —",
            font=ctk.CTkFont(size=15)
        )
        self.lbl_non_aml_conf.grid(row=0, column=1, padx=10, pady=5)

        self.warning_label = ctk.CTkLabel(
            right_frame,
            text="",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#FF6B6B",
            wraplength=350,
        )
        self.warning_label.grid(row=5, column=0, pady=(10, 5), padx=10)

        self.disclaimer_label = ctk.CTkLabel(
            right_frame,
            text="SCREENING FLAG FOR HUMAN REVIEW\nThis is not a clinical diagnosis.",
            font=ctk.CTkFont(size=12),
            text_color="#FFA500",
            wraplength=350,
        )
        self.disclaimer_label.grid(row=6, column=0, pady=(10, 15), padx=10)

        self.btn_analyze = ctk.CTkButton(
            right_frame, text="Run Analysis", command=self._run_analysis,
            width=200, height=40, font=ctk.CTkFont(size=14),
            fg_color="#2E7D32", hover_color="#1B5E20",
        )
        self.btn_analyze.grid(row=7, column=0, pady=(5, 15))
        self.btn_analyze.configure(state="disabled")

        # Status bar
        self.status_label = ctk.CTkLabel(
            self.tab_analyze, text=f"Ready. Load a blood smear image to begin. (AML threshold: {AML_THRESHOLD})",
            font=ctk.CTkFont(size=12), text_color="gray"
        )
        self.status_label.grid(row=8, column=0, columnspan=2, pady=(5, 10), sticky="w", padx=15)

        self.current_image_path = None

    def _build_history_tab(self):
        self.tab_history.grid_columnconfigure(0, weight=1)
        self.tab_history.grid_rowconfigure(0, weight=0)
        self.tab_history.grid_rowconfigure(1, weight=1)

        stats_frame = ctk.CTkFrame(self.tab_history)
        stats_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        stats_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.stat_total = ctk.CTkLabel(stats_frame, text="Total: 0", font=ctk.CTkFont(size=14, weight="bold"))
        self.stat_total.grid(row=0, column=0, padx=10, pady=5)
        self.stat_aml = ctk.CTkLabel(stats_frame, text="AML: 0", font=ctk.CTkFont(size=14))
        self.stat_aml.grid(row=0, column=1, padx=10, pady=5)
        self.stat_non = ctk.CTkLabel(stats_frame, text="Non-AML: 0", font=ctk.CTkFont(size=14))
        self.stat_non.grid(row=0, column=2, padx=10, pady=5)

        ctk.CTkButton(
            stats_frame, text="Refresh", command=self._refresh_history,
            width=100,
        ).grid(row=0, column=3, padx=10, pady=5)

        self.history_text = ctk.CTkTextbox(self.tab_history, font=ctk.CTkFont(size=12))
        self.history_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.history_text.configure(state="disabled")

    def _load_model(self):
        model_paths = [
            MODEL_DIR / "alnet_best.pt",
            MODEL_DIR / "alnet_final.pt",
            MODEL_DIR / "alnet_v2_best.pt",
            MODEL_DIR / "alnet_v2_final.pt",
        ]
        model_path = None
        for p in model_paths:
            if p.exists():
                model_path = p
                break

        if model_path is None:
            self.status_label.configure(text="ERROR: No model found in outputs/")
            return

        self.model = ALNet(num_classes=2).to(self.device)
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self.status_label.configure(
            text=f"Model loaded ({model_path.name}) | Device: {self.device} | Threshold: {AML_THRESHOLD} | Ready."
        )

    def _load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Blood Smear Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.current_image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((350, 350), Image.LANCZOS)

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.image_preview.configure(image=ctk_img, text="")

        self.btn_analyze.configure(state="normal")
        self.result_prediction.configure(text="—")
        self.result_confidence.configure(text="")
        self.progress_bar.set(0)
        self.lbl_aml_conf.configure(text="AML: —")
        self.lbl_non_aml_conf.configure(text="Non-AML: —")
        self.warning_label.configure(text="")
        self.status_label.configure(text=f"Loaded: {Path(file_path).name}")

    def _run_analysis(self):
        if self.current_image_path is None or self.model is None:
            return

        self.btn_analyze.configure(state="disabled", text="Analyzing...")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self.status_label.configure(text="Analyzing...")
        self.update_idletasks()

        try:
            img = Image.open(self.current_image_path).convert("RGB")
            img_tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

            aml_conf = float(probs[1])
            non_aml_conf = float(probs[0])
            pred_label = CLASS_NAMES[1] if aml_conf >= AML_THRESHOLD else CLASS_NAMES[0]
            confidence = aml_conf if pred_label == "AML" else non_aml_conf

            filename = Path(self.current_image_path).name
            log_prediction(filename, pred_label, [non_aml_conf, aml_conf])

            self._display_result(pred_label, confidence, aml_conf, non_aml_conf)

        except Exception as e:
            self._display_error(str(e))

    def _display_result(self, pred_label, confidence, aml_conf, non_aml_conf):
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(confidence)

        if pred_label == "AML":
            self.result_prediction.configure(text="AML FLAGGED (≥0.15)", text_color="#FF6B6B")
            self.warning_label.configure(
                text="FLAGGED — Refer for expert hematopathologist review."
            )
        else:
            self.result_prediction.configure(text="Non-AML", text_color="#4CAF50")
            self.warning_label.configure(text="")

        self.result_confidence.configure(
            text=f"Confidence: {confidence:.1%}",
            text_color="#FFFFFF" if confidence < 0.6 else ("#FF6B6B" if pred_label == "AML" else "#4CAF50"),
        )
        self.lbl_aml_conf.configure(text=f"AML: {aml_conf:.1%}")
        self.lbl_non_aml_conf.configure(text=f"Non-AML: {non_aml_conf:.1%}")

        self.btn_analyze.configure(state="normal", text="Run Analysis")
        self.status_label.configure(text="Analysis complete. Result logged.")

        self._refresh_history()

    def _display_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0)
        self.status_label.configure(text=f"ERROR: {error_msg}")
        self.btn_analyze.configure(state="normal", text="Run Analysis")

    def _refresh_history(self):
        stats = get_stats()
        self.stat_total.configure(text=f"Total: {stats['total']}")
        self.stat_aml.configure(text=f"AML: {stats['aml']}")
        self.stat_non.configure(text=f"Non-AML: {stats['non_aml']}")

        logs = get_recent_logs(limit=100)
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")

        header = f"{'Time':<22} {'Filename':<40} {'Result':<10} {'AML%':<8} {'Non-AML%'}\n"
        sep = "-" * 100 + "\n"
        self.history_text.insert("end", header + sep)

        for ts, fname, pred, aml_c, non_c in logs:
            short_fname = fname if len(fname) <= 38 else fname[:35] + "..."
            line = f"{ts[:19]:<22} {short_fname:<40} {pred:<10} {aml_c*100:>6.1f}% {non_c*100:>6.1f}%\n"
            self.history_text.insert("end", line)

        self.history_text.configure(state="disabled")


def main():
    init_db()
    app = ALNetDetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
