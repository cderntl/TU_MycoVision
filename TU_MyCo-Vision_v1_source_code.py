# ==================== TU_MyCo-Vision ===================
#
# Requirements:
# Python 3.10 - 3.12 recommended
# Written by: 
# Kartik Deopujari
# Matthias Schmal
# Good coding vibes by Chat GPT 
# =========================================================

import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import sys
import os
import time
import csv
import plotly.io as pio
import tempfile
import webbrowser
import plotly.express as px


try:
    from tkinterweb import HtmlFrame
except ImportError:
    HtmlFrame = None

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class MycoVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TU_MyCo-Vision: Fungal Morphology Detector")
        self.geometry("1300x900")
        self.configure(fg_color="#f5f7f0")
        self.resizable(True, True)

        # UI State
        self.input_folder = ctk.StringVar(value="")
        self.output_folder = ctk.StringVar(value="")
        self.selected_classes = [ctk.BooleanVar(value=False) for _ in range(13)]
        self.confidence = ctk.DoubleVar(value=0.2)
        self.line_width = ctk.IntVar(value=2)
        self.agnostic_nms = ctk.BooleanVar(value=False)
        self.show_output = ctk.BooleanVar(value=False)
        self.analysis_name = ctk.StringVar(value="Myco_Analysis")
        self.single_group_folder = ctk.StringVar(value="No folder selected")
        self.single_analysis_name = ctk.StringVar(value="SingleAnalysis")
        self.multi_analysis_name = ctk.StringVar(value="MultiAnalysis")
        self.conf_val_str = ctk.StringVar(value="0.20")
        self.group_folders = []
        self.group_display_var = ctk.StringVar(value="No groups selected")
        self.single_analysis_output_folder = ctk.StringVar(value="No folder selected")
        self.multi_analysis_output_folder = ctk.StringVar(value="No folder selected")
        self.excluded_classes = [ctk.BooleanVar(value=False) for _ in range(13)]
        self.class_names = ["C1", "C2", "C2-B", "C3-B", "C5", "C8", "C9", "C4", "C6", "C11", "C7-E", "C7-F", "C7-G"]
        self.yolo_to_class_index = [2, 0, 7, 8, 1, 9, 6, 3, 4, 5, 10, 11, 12]
        self.latest_prediction_folder = None
        self.last_analysis_csv = None
        self.analysis_results = {}

        self.bg_image_path = resource_path("aureo1.png")
        if os.path.exists(self.bg_image_path):
            self.bg_image = Image.open(self.bg_image_path).resize((1300, 850))
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            self.bg_label = ctk.CTkLabel(self, image=self.bg_photo, text="")
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.bg_label.lower()

        # --- Splash screen ---
        self.splash_frame = ctk.CTkFrame(self, fg_color="#ffffff", width=360, height=140, corner_radius=18, border_width=0)
        self.splash_frame.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(
            self.splash_frame,
            text="✨ Loading TU_MyCo-Vision...\nPlease wait...",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#168466",
            fg_color="transparent",
            justify="center"
        ).pack(pady=(22, 8))
        self.splash_progress = ctk.CTkProgressBar(
            self.splash_frame,
            height=10,
            corner_radius=5,
            progress_color="#21a179",
            fg_color="#dcdcdc"
        )
        self.splash_progress.pack(fill="x", padx=40, pady=(0, 16))
        self.splash_progress.set(0.1)
        self.after(200, self._start_heavy_imports)

    def _start_heavy_imports(self):
        def work():
            global pd, np, plt, cv2, Path, fitz, plotly, go
            import pandas as pd
            import numpy as np
            import cv2
            import scipy
            from pathlib import Path
            import plotly.graph_objs as go
            import plotly.io as pio
            self.splash_progress.set(0.7)
            time.sleep(0.5)
            self.splash_progress.set(1.0)
            self.after(400, self._show_main_ui)
        threading.Thread(target=work, daemon=True).start()

    def _show_main_ui(self):
        self.splash_frame.destroy()
        self._build_main_ui()

    def _build_main_ui(self):
        self.tabs = ctk.CTkTabview(self, width=1200, height=800)
        self.tabs.pack(expand=True, fill="both", padx=7, pady=6)
        self.panel_detection = self.tabs.add("Detection")
        self.panel_analysis = self.tabs.add("Analysis")
        self.panel_results = self.tabs.add("Results Dashboard")
        self._build_panel_detection(self.panel_detection)
        self._build_panel_analysis(self.panel_analysis)
        self._build_panel_results(self.panel_results)

    def clear_excluded_classes(self):
        for var in self.excluded_classes:
            var.set(False)
        

    # -----Detection Tab-----

    def _build_panel_detection(self, parent):
        parent.configure(fg_color="#f5f7f0")
        container = ctk.CTkFrame(parent, fg_color="#f5f7f0")
        container.pack(fill="both", expand=True, padx=20, pady=20)
        container.rowconfigure((0, 1, 2, 3), weight=1)
        container.columnconfigure((0, 1), weight=1)

        # 1. File I/O Block
        io_block = ctk.CTkFrame(container, fg_color="#eafaf1", border_width=1, border_color="#bbded6")
        io_block.grid(row=0, column=0, sticky="nsew", padx=6, pady=6, columnspan=2)
        io_block.columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(io_block, text="Select Image & Output Folder", font=ctk.CTkFont(size=16, weight="bold"), text_color="#388659").grid(row=0, column=0, columnspan=2, sticky="w", pady=(6, 3), padx=12)
        btn_img = ctk.CTkButton(io_block, text="Image Folder", command=self.browse_input, fg_color="#21a179")
        btn_img.grid(row=1, column=0, sticky="ew", padx=10, pady=8)
        btn_out = ctk.CTkButton(io_block, text="Output Folder", command=self.browse_output, fg_color="#406882")
        btn_out.grid(row=1, column=1, sticky="ew", padx=10, pady=8)
        path_frame_img = ctk.CTkFrame(io_block, fg_color="#f7fff7", border_width=1, border_color="#bbded6")
        path_frame_img.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        ctk.CTkLabel(path_frame_img, textvariable=self.input_folder, font=ctk.CTkFont(size=11), text_color="#457267").pack(fill="x", padx=6, pady=3)
        path_frame_out = ctk.CTkFrame(io_block, fg_color="#f7fff7", border_width=1, border_color="#bbded6")
        path_frame_out.grid(row=2, column=1, sticky="ew", padx=10, pady=(0, 10))
        ctk.CTkLabel(path_frame_out, textvariable=self.output_folder, font=ctk.CTkFont(size=11), text_color="#457267").pack(fill="x", padx=6, pady=3)

        # 2. Select Classes Block
        class_block = ctk.CTkFrame(container, fg_color="#eafaf1", border_width=1, border_color="#bbded6")
        class_block.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        class_block.columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkLabel(class_block, text="Select Classes", font=ctk.CTkFont(size=16, weight="bold"), text_color="#388659").grid(row=0, column=0, columnspan=3, pady=(5, 3))
        ctk.CTkButton(class_block, text="Select All", command=self.select_all_classes, fg_color="#d2f8d2", text_color="#388659").grid(row=1, column=0, padx=8, pady=4, sticky="ew")
        ctk.CTkButton(class_block, text="Clear Selection", command=self.clear_classes, fg_color="#fae588", text_color="#388659").grid(row=1, column=1, padx=8, pady=4, sticky="ew", columnspan=3)
        ctk.CTkButton(class_block, text="ℹ️Open Class Guide", width=10, height=10, fg_color="#fae588", text_color="#388659",
                      font=ctk.CTkFont(size=10, weight="bold"), command=self.show_class_reference_popup).grid(row=0, column=0, padx=(5, 5), pady=(5, 0), sticky="w")
        class_names = ["C2", "C5", "C1", "C4", "C6", "C11", "C9", "C2-B", "C3-B", "C8", "C7-E", "C7-F", "C7-G"]
        for i, cname in enumerate(class_names):
            row_i = 2 + i // 4
            col_i = i % 4
            cb = ctk.CTkCheckBox(class_block, text=cname, variable=self.selected_classes[i], font=ctk.CTkFont(size=12), fg_color="#388659")
            cb.grid(row=row_i, column=col_i, padx=4, pady=4, sticky="w")

        # 3. Model Settings Block
        model_block = ctk.CTkFrame(container, fg_color="#eafaf1", border_width=1, border_color="#bbded6")
        model_block.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        model_block.columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(model_block, text="Model Settings", font=ctk.CTkFont(size=16, weight="bold"), text_color="#406882").grid(row=0, column=0, columnspan=2, pady=(6, 3))
        ctk.CTkLabel(model_block, text="Confidence:", font=ctk.CTkFont(size=12)).grid(row=1, column=0, padx=8, pady=3, sticky="e")
        conf_slider = ctk.CTkSlider(model_block, from_=0.1, to=1.0, variable=self.confidence, width=120, progress_color="#21a179", button_color="#21a179", number_of_steps=18, command=self.update_conf_val)
        conf_slider.grid(row=1, column=1, padx=8, pady=3, sticky="w")
        ctk.CTkLabel(model_block, textvariable=self.conf_val_str, font=ctk.CTkFont(size=12, weight="bold")).grid(row=2, column=1, sticky="w")
        ctk.CTkLabel(model_block, text="Line Width:", font=ctk.CTkFont(size=12)).grid(row=3, column=0, padx=8, pady=3, sticky="e")
        ctk.CTkEntry(model_block, textvariable=self.line_width, width=44).grid(row=3, column=1, padx=8, pady=3, sticky="w")
        ctk.CTkLabel(model_block, text="Analysis Name:", font=ctk.CTkFont(size=12)).grid(row=4, column=0, padx=8, pady=3, sticky="e")
        ctk.CTkEntry(model_block, textvariable=self.analysis_name, width=100).grid(row=4, column=1, padx=8, pady=3, sticky="w")
        ctk.CTkCheckBox(model_block, text="Agnostic NMS", variable=self.agnostic_nms, fg_color="#21a179").grid(row=5, column=0, padx=8, pady=3, sticky="w")
        ctk.CTkCheckBox(model_block, text="Show Output", variable=self.show_output, fg_color="#21a179").grid(row=5, column=1, padx=8, pady=3, sticky="w")
        self.update_conf_val(self.confidence.get())

        # 4. Progress Bar & Predict Button
        self.task_progress = ctk.CTkProgressBar(container, height=15, corner_radius=6, progress_color="#00589C")
        self.task_progress.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 4))
        self.task_progress.set(0.0)

        predict_block = ctk.CTkFrame(container, fg_color="#f5f7f0")
        predict_block.grid(row=3, column=0, columnspan=2, padx=6, pady=8, sticky="ew")
        predict_block.columnconfigure(0, weight=1)
        btn_pred = ctk.CTkButton(predict_block, text="▶️ Run MyCo-Vision Prediction", command=self.run_prediction_threaded,
                                 font=ctk.CTkFont(size=20, weight="bold"), fg_color="#21a179", text_color="#fff", height=60, width=500)
        btn_pred.grid(row=0, column=0, pady=12, sticky="ew")

    # ------------- Helper Functions for Detection Tab -------------
    def browse_input(self):
        path = filedialog.askdirectory(title="Select Image Folder")
        if path:
            self.input_folder.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_folder.set(path)

    def update_conf_val(self, value):
        self.conf_val_str.set(f"{float(value):.2f}")

    def clear_classes(self):
        for var in self.selected_classes:
            var.set(False)

    def select_all_classes(self):
        for var in self.selected_classes:
            var.set(True)

    def show_class_reference_popup(self):
        img_path = resource_path("reference.png")
        if not os.path.exists(img_path):
            messagebox.showerror("Image Not Found", f"Reference image not found:\n{img_path}")
            return
        popup = ctk.CTkToplevel(self)
        popup.title("Morphotype Class Reference")
        popup.geometry("800x600")
        popup.lift()
        popup.focus_force()
        popup.grab_set()
        frame = ctk.CTkFrame(popup)
        frame.pack(fill="both", expand=True)
        canvas = ctk.CTkCanvas(frame, bg="white", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        img = Image.open(img_path)
        def draw_img(event=None):
            w, h = canvas.winfo_width(), canvas.winfo_height()
            if w < 10 or h < 10:
                return
            scaled = img.copy()
            scaled.thumbnail((w - 20, h - 20))
            canvas.image = ImageTk.PhotoImage(scaled)
            canvas.delete("all")
            canvas.create_image(w // 2, h // 2, anchor="center", image=canvas.image)
        canvas.bind("<Configure>", draw_img)
        draw_img()
        popup.focus_set()
        
    # ----- Prediction Function ------

    def run_prediction_threaded(self):
        threading.Thread(target=self._run_prediction, daemon=True).start()

    def _run_prediction(self):
        self.task_progress.set(0.05)
        try:
            from ultralytics import YOLO
            import cv2

            model_path = resource_path('TU_MyCo-Vision.pt')
            model = YOLO(model_path)
            selected_cls = [i for i, v in enumerate(self.selected_classes) if v.get()]
            if not self.input_folder.get() or not self.output_folder.get():
                messagebox.showerror("Input Error", "Please select input and output folders.")
                return
            if not selected_cls:
                messagebox.showerror("Input Error", "Please select at least one class.")
                return
            img_folder = self.input_folder.get()
            imgsz = (640, 640)
            for f in os.listdir(img_folder):
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
                    img = cv2.imread(os.path.join(img_folder, f))
                    if img is not None:
                        h, w = img.shape[:2]
                        imgsz = (max(32, round(h / 32) * 32), max(32, round(w / 32) * 32))
                        break

            model.predict(
                source=img_folder,
                imgsz=imgsz,
                conf=self.confidence.get(),
                line_width=self.line_width.get(),
                classes=selected_cls,
                save=True, save_txt=True, save_conf=True,
                agnostic_nms=self.agnostic_nms.get(),
                augment=True, 
                show=self.show_output.get(),
                name=self.analysis_name.get(),
                project=self.output_folder.get(),
                max_det=1000
            )
            self.task_progress.set(1.0)
            messagebox.showinfo("Done", "YOLO Prediction completed!")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
        finally:
            self.task_progress.set(0.0)

    # ----- Analysis Tab -----
    def _build_panel_analysis(self, parent):
        parent.configure(fg_color="#f5f7f0")
        container = ctk.CTkFrame(parent, fg_color="#f5f7f0")
        container.pack(fill="both", expand=True, padx=20, pady=20)
        container.rowconfigure((0, 1, 2, 3), weight=1)
        container.columnconfigure((0, 1), weight=1)


        # Exclusion Panel
        excl_block = ctk.CTkFrame(container, fg_color="#eafaf1", border_width=1, border_color="#bbded6")
        excl_block.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        excl_block.columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkButton(excl_block, text="Clear Selection",command=self.clear_excluded_classes, fg_color="#fae588", text_color="#388659").grid(row=0, column=3, padx=8, pady=(6,3), sticky="e")
        ctk.CTkLabel(excl_block, text="Class Exclusion (Exclude & Merge Selected Classes as ‘Others’)", font=ctk.CTkFont(size=15, weight="bold"), text_color="#388659").grid(row=0, column=0, columnspan=4, sticky="w", pady=(6, 3), padx=12)
        for i, cname in enumerate(self.class_names):
            row_i = 1 + i // 4
            col_i = i % 4
            cb = ctk.CTkCheckBox(excl_block, text=cname, variable=self.excluded_classes[i], font=ctk.CTkFont(size=12))
            cb.grid(row=row_i, column=col_i, padx=6, pady=4, sticky="w")
            

        # --- Single Group Analysis Block ---
        sg_frame = ctk.CTkFrame(container, fg_color="#eafaf1", border_width=1, border_color="#bbded6")
        sg_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(sg_frame, text="Single Group Analysis", font=ctk.CTkFont(size=14, weight="bold"), text_color="#388659").grid(row=0, column=0, columnspan=2, sticky="w", pady=(5, 2))
        ctk.CTkButton(sg_frame, text="Select Folder", command=self.browse_single_group_folder, fg_color="#21a179").grid(row=1, column=0, padx=8, pady=4, sticky="w")
        sg_folder_panel = ctk.CTkFrame(sg_frame, fg_color="#f7fff7", border_width=1, border_color="#bbded6")
        sg_folder_panel.grid(row=1, column=1, padx=8, pady=4, sticky="ew")
        ctk.CTkLabel(sg_folder_panel, textvariable=self.single_group_folder, font=ctk.CTkFont(size=11), text_color="#457267", anchor="w").pack(fill="x", padx=4, pady=2)
        ctk.CTkButton(sg_frame, text="Select Output Folder", command=self.select_single_analysis_output_folder, fg_color="#406882").grid(row=2, column=0, padx=8, pady=4, sticky="w")
        sg_out_panel = ctk.CTkFrame(sg_frame, fg_color="#f7fff7", border_width=1, border_color="#bbded6")
        sg_out_panel.grid(row=2, column=1, padx=8, pady=4, sticky="ew")
        ctk.CTkLabel(sg_out_panel, textvariable=self.single_analysis_output_folder, font=ctk.CTkFont(size=11), text_color="#457267", anchor="w").pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(sg_frame, text="Analysis Name:", font=ctk.CTkFont(size=12)).grid(row=3, column=0, padx=8, pady=4, sticky="w")
        ctk.CTkEntry(sg_frame, textvariable=self.single_analysis_name, width=120).grid(row=3, column=1, padx=8, pady=4, sticky="w")
        ctk.CTkButton(sg_frame, text="Run Single Group Analysis", command=self.run_single_group_analysis_threaded, font=ctk.CTkFont(size=14, weight="bold"), fg_color="#388659", text_color="#fff").grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        # --- Multi Group Analysis Block ---
        mg_frame = ctk.CTkFrame(container, fg_color="#eafaf1", border_width=1, border_color="#bbded6")
        mg_frame.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        ctk.CTkLabel(mg_frame, text="Multi Group Analysis", font=ctk.CTkFont(size=14, weight="bold"), text_color="#406882").grid(row=0, column=0, columnspan=2, sticky="w", pady=(5, 2))
        ctk.CTkButton(mg_frame, text="Add Group Folders", command=self.add_group_folder, fg_color="#21a179").grid(row=1, column=0, padx=8, pady=4, sticky="w")
        ctk.CTkButton(mg_frame, text="Clear Groups", command=self.clear_group_folders, fg_color="#d2f8d2", text_color="#388659").grid(row=1, column=1, padx=8, pady=4, sticky="w")
        ctk.CTkLabel(mg_frame, text="Selected Groups:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#388659", anchor="w").grid(row=2, column=0, padx=8, pady=4, sticky="nw")
        mg_group_panel = ctk.CTkFrame(mg_frame, fg_color="#f7fff7", border_width=1, border_color="#bbded6")
        mg_group_panel.grid(row=2, column=1, padx=8, pady=4, sticky="ew")
        ctk.CTkLabel(mg_group_panel, textvariable=self.group_display_var, font=ctk.CTkFont(size=11), text_color="#457267", anchor="nw", justify="left", wraplength=180).pack(fill="x", padx=4, pady=2)
        ctk.CTkButton(mg_frame, text="Select Output Folder", command=self.select_multi_analysis_output_folder, fg_color="#406882").grid(row=3, column=0, padx=8, pady=4, sticky="w")
        mg_out_panel = ctk.CTkFrame(mg_frame, fg_color="#f7fff7", border_width=1, border_color="#bbded6")
        mg_out_panel.grid(row=3, column=1, padx=8, pady=4, sticky="ew")
        ctk.CTkLabel(mg_out_panel, textvariable=self.multi_analysis_output_folder, font=ctk.CTkFont(size=11), text_color="#457267", anchor="w").pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(mg_frame, text="Analysis Name:", font=ctk.CTkFont(size=12)).grid(row=4, column=0, padx=8, pady=4, sticky="w")
        ctk.CTkEntry(mg_frame, textvariable=self.multi_analysis_name, width=120).grid(row=4, column=1, padx=8, pady=4, sticky="w")
        ctk.CTkButton(mg_frame, text="Run Multi Group Analysis", command=self.run_multi_group_analysis_threaded, font=ctk.CTkFont(size=14, weight="bold"), fg_color="#406882", text_color="#fff").grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

    # === Analysis panel helpers ===
    def browse_single_group_folder(self):
        path = filedialog.askdirectory(title="Select Folder for Single Group Analysis")
        if path:
            self.single_group_folder.set(path)
    def select_single_analysis_output_folder(self):
        path = filedialog.askdirectory(title="Select Output Folder for Single Group Analysis")
        if path:
            self.single_analysis_output_folder.set(path)
    def select_multi_analysis_output_folder(self):
        path = filedialog.askdirectory(title="Select Output Folder for Multi Group Analysis")
        if path:
            self.multi_analysis_output_folder.set(path)
    def add_group_folder(self):
        path = filedialog.askdirectory(title="Add Group Folder for Multi-Group Analysis")
        if path and path not in self.group_folders:
            self.group_folders.append(path)
            self.group_display_var.set("\n".join([os.path.basename(p) for p in self.group_folders]))
    def clear_group_folders(self):
        self.group_folders = []
        self.group_display_var.set("No groups selected")
    
    #----- Single & Multi-Group Analysis (logics)-----

    def run_single_group_analysis_threaded(self):
        threading.Thread(target=self._single_group_analysis, daemon=True).start()

    def run_multi_group_analysis_threaded(self):
        threading.Thread(target=self._multi_group_analysis, daemon=True).start()

    def _single_group_analysis(self):
        import pandas as pd
        import numpy as np
        from pathlib import Path
        self.task_progress.set(0.1)
        classes_all = ["C1", "C2", "C2-B", "C3-B", "C5", "C8", "C9", "C4", "C6", "C11", "C7-E", "C7-F", "C7-G"]
        correct_all = [2, 0, 7, 8, 1, 9, 6, 3, 4, 5, 10, 11, 12]
        epsilon = 0.001

        excluded_indices = [i for i, var in enumerate(self.excluded_classes) if var.get()]
        included_indices = [i for i in range(13) if i not in excluded_indices]

        folder = self.single_group_folder.get()
        label_dir = Path(folder) / "labels"
        if not label_dir.exists():
            label_dir = Path(folder)
        label_files = sorted(label_dir.glob("*.txt"))
        if not label_files:
            messagebox.showerror("Error", "No label .txt files found in selected folder.")
            self.task_progress.set(0.0)
            return

        image_names, all_counts_raw = [], []
        for idx, path in enumerate(label_files):
            temp_counts = [0] * 13
            with open(path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) < 6: continue
                    values = [float(x) for x in parts]
                    marginx = [abs((values[1] + values[3]/2) - 1) < epsilon, (values[1] - values[3]/2) < epsilon]
                    marginy = [abs((values[2] + values[4]/2) - 1) < epsilon, (values[2] - values[4]/2) < epsilon]
                    if (sum(marginx) or sum(marginy)) and int(values[0]) not in [4, 10, 11, 12]: continue
                    temp_counts[int(values[0])] += 1
            ordered = [0] * 13
            for i, j in enumerate(correct_all): ordered[i] = temp_counts[j]
            all_counts_raw.append(ordered)
            image_names.append(path.stem)
            self.task_progress.set(0.1 + 0.6 * (idx+1)/len(label_files))

        analysis_name = self.single_analysis_name.get()
        analysis_save_dir = self.single_analysis_output_folder.get()
        if not analysis_save_dir or analysis_save_dir == "No folder selected":
            analysis_save_dir = self.output_folder.get() or str(label_dir.parent)
        save_dir = Path(analysis_save_dir)

        df_raw = pd.DataFrame(all_counts_raw, columns=classes_all)
        df_raw.insert(0, "Image", image_names)
        df_raw.to_csv(save_dir / f"{analysis_name}_rawdata.csv", index=False)
        self.last_analysis_csv = str(save_dir / f"{analysis_name}_rawdata.csv")

        df_mod = df_raw.copy()
        if excluded_indices:
            df_mod["Others"] = df_mod[[classes_all[i] for i in excluded_indices]].sum(axis=1)
            df_mod.drop([classes_all[i] for i in excluded_indices], axis=1, inplace=True)
            df_mod.to_csv(save_dir / f"{analysis_name}_excluded.csv", index=False)


        self.task_progress.set(1.0)
        messagebox.showinfo("Done", f"Single Group CSVs saved in:\n{save_dir}")

        # Update dashboard viewer after analysis
        try:
            import pandas as pd
            df_dashboard = pd.read_csv(self.last_analysis_csv)
            self.dashboard_loaded_images_dir = folder
            self._dashboard_refresh_image_pairs()
        except Exception as e:
            print("Dashboard auto-refresh failed:", e)

    def _multi_group_analysis(self):
        import pandas as pd
        import numpy as np
        from pathlib import Path

        self.task_progress.set(0.1)
        classes_all = ["C1", "C2", "C2-B", "C3-B", "C5", "C8", "C9", "C4", "C6", "C11", "C7-E", "C7-F", "C7-G"]
        correct_all = [2, 0, 7, 8, 1, 9, 6, 3, 4, 5, 10, 11, 12]
        epsilon = 0.001

        excluded = [i for i, var in enumerate(self.excluded_classes) if var.get()]

        all_counts, group_labels, image_labels = [], [], []
        folders = self.group_folders
        for g_idx, folder in enumerate(folders):
            label_dir = Path(folder) / "labels"
            if not label_dir.exists():
                label_dir = Path(folder)
            files = sorted(label_dir.glob("*.txt"))
            if not files:
                continue
            img_counts = []
            for p_idx, path in enumerate(files):
                counts = [0] * 13
                with open(path) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 6:
                            continue
                        vals = list(map(float, parts))
                        mx = abs((vals[1] + vals[3]/2) - 1) < epsilon or (vals[1] - vals[3]/2) < epsilon
                        my = abs((vals[2] + vals[4]/2) - 1) < epsilon or (vals[2] - vals[4]/2) < epsilon
                        if (mx or my) and int(vals[0]) not in [4, 10, 11, 12]:
                            continue
                        counts[int(vals[0])] += 1
                ordered = [counts[j] for j in correct_all]
                img_counts.append(ordered)
                all_counts.append(ordered)
                group_labels.append(Path(folder).name)
                image_labels.append(path.stem)
            self.task_progress.set(0.1 + 0.8*(g_idx+1)/len(folders))

        analysis_name = self.multi_analysis_name.get()
        save_dir = Path(self.multi_analysis_output_folder.get() or self.output_folder.get() or str(label_dir.parent))

        df_all = pd.DataFrame(all_counts, columns=classes_all)
        df_all['Group'] = group_labels
        df_all['Image'] = image_labels
        df_all.to_csv(save_dir / f"{analysis_name}_rawdata.csv", index=False)
        self.last_analysis_csv = str(save_dir / f"{analysis_name}_rawdata.csv")

        df_mod = df_all.copy()
        if excluded:
            df_mod['Others'] = df_mod[[classes_all[i] for i in excluded]].sum(axis=1)
            df_mod.drop(columns=[classes_all[i] for i in excluded], inplace=True)
            df_mod.to_csv(save_dir / f"{analysis_name}_excluded.csv", index=False)

        self.task_progress.set(1.0)
        messagebox.showinfo('Done', f'Multi-Group CSVs saved in:\n{save_dir}')
        # Update dashboard viewer after analysis
        try:
            import pandas as pd
            df_dashboard = pd.read_csv(self.last_analysis_csv)
            self.dashboard_loaded_images_dir = folders[0] if folders else ""
            self._dashboard_refresh_image_pairs()
        except Exception as e:
            print("Dashboard auto-refresh failed:", e)

    # ----- Results Dashboard Panel -----
    def set_dashboard_folder(self, which):
        path = filedialog.askdirectory(title=f"Select {'Original' if which == 'orig' else 'Predicted'} Images Folder")
        if path:
            if which == "orig":
                self.orig_img_folder.set(path)
            else:
                self.pred_img_folder.set(path)
            # Refresh the image viewer panel with new folder(s)
            if hasattr(self, "dashboard_images_panel") and self.dashboard_images_panel:
                self._dashboard_build_image_viewer(self.dashboard_images_panel)


    
    def _build_panel_results(self, parent):
        parent.configure(fg_color="#f5f7f0")
        main_frame = ctk.CTkFrame(parent, fg_color="#f5f7f0")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        btnrow = ctk.CTkFrame(main_frame, fg_color="#f5f7f0")
        btnrow.pack(fill="x", padx=10, pady=(4, 0))
        ctk.CTkButton(btnrow, text="Load Analysis CSV", command=self.dashboard_load_csv, fg_color="#388659").pack(side="left", padx=5)
        ctk.CTkButton(btnrow, text="View CSV Table", command=self.dashboard_view_csv_table, fg_color="#406882").pack(side="left", padx=5)
        ctk.CTkButton(btnrow, text="Click to View Plots", command=self.dashboard_view_plots, fg_color="#B8860B").pack(side="left", padx=5)

        right_panel = ctk.CTkFrame(btnrow, fg_color="#f5f7f0")
        right_panel.pack(side="right", padx=5)
        self.orig_img_folder = ctk.StringVar(value="")
        self.pred_img_folder = ctk.StringVar(value="")
        ctk.CTkButton(right_panel, text="Original Images...", fg_color="#30cc7c",
                    command=lambda: self.set_dashboard_folder("orig")).pack(side="left", padx=4)
        ctk.CTkLabel(right_panel, textvariable=self.orig_img_folder, width=160, anchor="w", text_color="#222", fg_color="#ebfaeb").pack(side="left", padx=(0, 4))
        ctk.CTkButton(right_panel, text="Predicted Images...", fg_color="#1eb2e6",
                    command=lambda: self.set_dashboard_folder("pred")).pack(side="left", padx=4)
        ctk.CTkLabel(right_panel, textvariable=self.pred_img_folder, width=160, anchor="w", text_color="#222", fg_color="#e6f4fa").pack(side="left", padx=(0, 4))

        # Main area: by default show results panel, not image viewer
        self.results_or_images = ctk.StringVar(value="results")
        tabs = ctk.CTkTabview(main_frame, width=1180, height=670)
        tabs.pack(fill="both", expand=True, padx=6, pady=(8,6))
        results_panel = tabs.add("Results")
        images_panel = tabs.add("Image Viewer")
        self.dashboard_images_panel = images_panel
        self._dashboard_build_results_panel(results_panel)
        self._dashboard_build_image_viewer(images_panel)
        tabs.set("Results")  # Always default to results

    def _dashboard_build_results_panel(self, parent):
        ctk.CTkLabel(parent, text="Click 'Button --> Click to View Plots' to view interactive results.", font=ctk.CTkFont(size=18, weight="bold"), text_color="#406882").pack(pady=24)

    def dashboard_load_csv(self):
        import pandas as pd
        path = filedialog.askopenfilename(title="Select Analysis CSV", filetypes=[("CSV Files", "*.csv")])
        if not path or not os.path.exists(path):
            return
        self.last_analysis_csv = path

    def dashboard_view_csv_table(self):
        # Reuse existing logic: show a table window
        if not self.last_analysis_csv or not os.path.exists(self.last_analysis_csv):
            messagebox.showinfo("No Data", "No analysis CSV loaded.")
            return
        self.open_csv_table_path(self.last_analysis_csv)

    def dashboard_view_plots(self):
        # Generate and open plots in browser (actual plotting code comes next block)
        if not self.last_analysis_csv or not os.path.exists(self.last_analysis_csv):
            messagebox.showinfo("No Data", "No analysis CSV loaded.")
            return
        try:
            self._dashboard_generate_and_open_plots(self.last_analysis_csv)
        except Exception as e:
            messagebox.showerror("Plot Error", f"Error creating plots: {e}")

    def open_csv_table_path(self, path):
        from tkinter import Toplevel
        import pandas as pd
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("CSV Error", f"Could not read CSV: {e}")
            return
        win = Toplevel(self)
        win.title("CSV Data Table")
        win.geometry("1000x700")
        tree = ttk.Treeview(win, show="headings")
        tree.pack(fill="both", expand=True)
        tree["columns"] = list(df.columns)
        for c in df.columns:
            tree.heading(c, text=c)
            tree.column(c, anchor="center", width=90)
        for i, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        ctk.CTkButton(win, text="Export CSV", command=lambda: self.export_csv_table(df)).pack(pady=10)

    def export_csv_table(self, df):
        path = filedialog.asksaveasfilename(title="Export CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            df.to_csv(path, index=False)
            messagebox.showinfo("Exported", f"CSV exported to:\n{path}")

    # --- IMAGE VIEWER: always present, split panel, main dashboard area ---
    def _dashboard_build_image_viewer(self, parent):
        import glob
        from PIL import Image, ImageTk
        from tkinter import Canvas, HORIZONTAL, Scrollbar

        # Remove old children
        for child in parent.winfo_children():
            child.destroy()

        viewer_main = ctk.CTkFrame(parent, fg_color="#fff")
        viewer_main.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        viewer_main.grid_columnconfigure(0, weight=1)
        viewer_main.grid_columnconfigure(1, weight=1)
        viewer_main.grid_rowconfigure(1, weight=1)  # The canvases

        self.left_img_label = ctk.CTkLabel(viewer_main, text="", font=ctk.CTkFont(size=16), text_color="#2a5d7c")
        self.right_img_label = ctk.CTkLabel(viewer_main, text="", font=ctk.CTkFont(size=16), text_color="#2a5d7c")
        self.left_img_label.grid(row=0, column=0, pady=(2,0))
        self.right_img_label.grid(row=0, column=1, pady=(2,0))

        left_canvas = Canvas(viewer_main, bg="white", width=520, height=440, highlightthickness=0)
        right_canvas = Canvas(viewer_main, bg="white", width=520, height=440, highlightthickness=0)
        left_canvas.grid(row=1, column=0, padx=10, pady=12, sticky="nsew")
        right_canvas.grid(row=1, column=1, padx=10, pady=12, sticky="nsew")

        # Create label widgets for filenames
        self.left_img_label = ctk.CTkLabel(viewer_main, text="", font=ctk.CTkFont(size=12), text_color="#2a5d7c")
        self.right_img_label = ctk.CTkLabel(viewer_main, text="", font=ctk.CTkFont(size=12), text_color="#2a5d7c")
        self.left_img_label.grid(row=0, column=0, pady=(4,0))
        self.right_img_label.grid(row=0, column=1, pady=(4,0))


        # --- Two Canvases: Original (left), Predicted (right) ---
        left_canvas = Canvas(viewer_main, bg="white", highlightthickness=0)
        right_canvas = Canvas(viewer_main, bg="white", highlightthickness=0)
        left_canvas.grid(row=1, column=0, padx=10, pady=12, sticky="nsew")
        right_canvas.grid(row=1, column=1, padx=10, pady=12, sticky="nsew")
        viewer_main.grid_columnconfigure(0, weight=1)
        viewer_main.grid_columnconfigure(1, weight=1)
        viewer_main.grid_rowconfigure(0, weight=1)

        # --- Navigation Buttons & Counter ---
        nav_panel = ctk.CTkFrame(parent, fg_color="#f5f7f0")
        nav_panel.pack(fill="x", padx=24, pady=(0, 0))
        btn_prev = ctk.CTkButton(nav_panel, text="⟨ Prev", width=80)
        btn_prev.pack(side="left", padx=(0, 18))
        btn_next = ctk.CTkButton(nav_panel, text="Next ⟩", width=80)
        btn_next.pack(side="left")
        image_counter_label = ctk.CTkLabel(nav_panel, text="", font=ctk.CTkFont(size=14))
        image_counter_label.pack(side="left", padx=18)
        self.dashboard_image_counter_label = image_counter_label  # for external update if needed

        # --- Thumbnails Bar ---
        thumb_outer = ctk.CTkFrame(parent, fg_color="#f3f3f8")
        thumb_outer.pack(fill="x", padx=2, pady=(0, 6))
        thumb_canvas = Canvas(thumb_outer, height=90, bg="#f3f3f8", highlightthickness=0)
        thumb_scroll = Scrollbar(thumb_outer, orient=HORIZONTAL, command=thumb_canvas.xview)
        thumb_scroll.pack(side="bottom", fill="x")
        thumb_canvas.pack(side="top", fill="x", expand=True)
        thumb_canvas.configure(xscrollcommand=thumb_scroll.set)
        thumb_bar = ctk.CTkFrame(thumb_canvas, fg_color="#f3f3f8")
        thumb_canvas.create_window((0, 0), window=thumb_bar, anchor="nw")
        thumb_bar.bind("<Configure>", lambda e: thumb_canvas.configure(scrollregion=thumb_canvas.bbox("all")))

        # --- Image matching and loading ---
        orig_dir = self.orig_img_folder.get() if hasattr(self, "orig_img_folder") else ""
        pred_dir = self.pred_img_folder.get() if hasattr(self, "pred_img_folder") else ""
        if not orig_dir or not os.path.exists(orig_dir) or not pred_dir or not os.path.exists(pred_dir):
            left_canvas.create_text(260, 220, text="Select both original and predicted image folders above!", fill="gray", font=("Arial", 13))
            right_canvas.create_text(260, 220, text="No images", fill="gray", font=("Arial", 13))
            image_counter_label.configure(text="")
            return

        orig_files = [p for p in glob.glob(os.path.join(orig_dir, "*")) if p.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))]
        pred_files = [p for p in glob.glob(os.path.join(pred_dir, "*")) if p.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))]
        orig_by_base = {os.path.splitext(os.path.basename(p))[0]: p for p in orig_files}
        pred_by_base = {os.path.splitext(os.path.basename(p))[0]: p for p in pred_files}
        matches = sorted(set(orig_by_base.keys()) & set(pred_by_base.keys()))
        if not matches:
            left_canvas.create_text(260, 220, text="No matching basenames found between folders.", fill="gray", font=("Arial", 13))
            right_canvas.create_text(260, 220, text="No matches.", fill="gray", font=("Arial", 13))
            image_counter_label.configure(text="")
            return

        # Viewer state per image
        state = {
            "idx": 0,
            "matches": matches,
            "orig": orig_by_base,
            "pred": pred_by_base,
            "imgL": None,
            "imgR": None,
            "tkL": None,
            "tkR": None,
            "zoomL": 1.0,
            "zoomR": 1.0,
            "offsetL": [0, 0],
            "offsetR": [0, 0],
            "dragL": None,
            "dragR": None,
            "fitL": 1.0,
            "fitR": 1.0,
        }

        def fit_scale(img, can_w, can_h):
            iw, ih = img.size
            if iw == 0 or ih == 0:
                return 1.0
            return min(can_w / iw, can_h / ih)

        def show(idx, resize_only=False):
            base = state["matches"][idx]
            orig_name = os.path.basename(state["orig"][base])
            pred_name = os.path.basename(state["pred"][base])
            self.left_img_label.configure(text=f"Original: {orig_name}")
            self.right_img_label.configure(text=f"Predicted: {pred_name}")
            img_o = Image.open(state["orig"][base]).convert("RGB")
            img_p = Image.open(state["pred"][base]).convert("RGB")
            state["imgL"], state["imgR"] = img_o, img_p

            can_w = left_canvas.winfo_width() or 520
            can_h = left_canvas.winfo_height() or 440
            fitL = fit_scale(img_o, can_w, can_h)
            fitR = fit_scale(img_p, can_w, can_h)
            state["fitL"] = fitL
            state["fitR"] = fitR

            if not resize_only:
                state["zoomL"], state["zoomR"] = fitL, fitR
                state["offsetL"], state["offsetR"] = [0, 0], [0, 0]
            draw_canvas(left_canvas, "L")
            draw_canvas(right_canvas, "R")
            image_counter_label.configure(text=f"Image {idx+1} / {len(state['matches'])}")

        def draw_canvas(canvas, side="L"):
            img = state["imgL"] if side == "L" else state["imgR"]
            zoom = state["zoomL"] if side == "L" else state["zoomR"]
            offset = state["offsetL"] if side == "L" else state["offsetR"]
            can_w = max(canvas.winfo_width(), 1)
            can_h = max(canvas.winfo_height(), 1)
            iw, ih = img.size

            fit_zoom = min(can_w / iw, can_h / ih)
            # Clamp zoom to at least fit, at most 10x
            if zoom < fit_zoom:
                zoom = fit_zoom
                if side == "L":
                    state["zoomL"] = zoom
                else:
                    state["zoomR"] = zoom
            elif zoom > fit_zoom * 10:
                zoom = fit_zoom * 10
                if side == "L":
                    state["zoomL"] = zoom
                else:
                    state["zoomR"] = zoom

            new_w, new_h = int(iw * zoom), int(ih * zoom)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)

            # Center if not panned
            if new_w <= can_w:
                x0 = (can_w - new_w) // 2
                offset[0] = 0
            else:
                x0 = -offset[0]
                offset[0] = min(max(offset[0], 0), new_w - can_w)
            if new_h <= can_h:
                y0 = (can_h - new_h) // 2
                offset[1] = 0
            else:
                y0 = -offset[1]
                offset[1] = min(max(offset[1], 0), new_h - can_h)

            canvas.delete("all")
            canvas.create_rectangle(0, 0, can_w, can_h, fill="white", outline="white")
            tkimg = ImageTk.PhotoImage(img_resized)
            canvas.image = tkimg
            canvas.create_image(x0 + new_w // 2, y0 + new_h // 2, image=tkimg, anchor="center")
            fit = state["fitL"] if side == "L" else state["fitR"]
            zoom_txt = f"Zoom: {zoom/fit:.2f}x"
            canvas.create_text(can_w//2, can_h-18, text=zoom_txt, fill="#388659", font=("Arial", 12))

        def zoom(event, side="L"):
            if side == "L":
                zoom = state["zoomL"]
                canvas = left_canvas
                fit = state["fitL"]
                offset = state["offsetL"]
            else:
                zoom = state["zoomR"]
                canvas = right_canvas
                fit = state["fitR"]
                offset = state["offsetR"]
            # Wheel up = zoom in, down = zoom out
            if (hasattr(event, 'delta') and event.delta > 0) or getattr(event, "num", None) == 4:
                factor = 1.13
            else:
                factor = 0.88
            new_zoom = max(fit, min(zoom * factor, fit * 10.0))
            # Adjust offset to center on mouse (relative to image)
            can_w, can_h = canvas.winfo_width(), canvas.winfo_height()
            mouse_x, mouse_y = event.x, event.y
            rel_x = (offset[0] + mouse_x) / zoom
            rel_y = (offset[1] + mouse_y) / zoom
            offset[0] = int(rel_x * new_zoom - mouse_x)
            offset[1] = int(rel_y * new_zoom - mouse_y)
            if side == "L":
                state["zoomL"] = new_zoom
            else:
                state["zoomR"] = new_zoom
            draw_canvas(canvas, side)

        def drag_start(event, side="L"):
            if side == "L":
                state["dragL"] = (event.x, event.y)
            else:
                state["dragR"] = (event.x, event.y)

        def drag_motion(event, side="L"):
            if side == "L":
                dx = event.x - state["dragL"][0]
                dy = event.y - state["dragL"][1]
                state["offsetL"][0] -= dx
                state["offsetL"][1] -= dy
                state["dragL"] = (event.x, event.y)
                draw_canvas(left_canvas, "L")
            else:
                dx = event.x - state["dragR"][0]
                dy = event.y - state["dragR"][1]
                state["offsetR"][0] -= dx
                state["offsetR"][1] -= dy
                state["dragR"] = (event.x, event.y)
                draw_canvas(right_canvas, "R")

        # Bindings for zoom (wheel), pan (drag)
        for canvas, side in [(left_canvas, "L"), (right_canvas, "R")]:
            canvas.bind("<MouseWheel>", lambda e, s=side: zoom(e, s))
            canvas.bind("<Button-4>", lambda e, s=side: zoom(e, s))  # Linux
            canvas.bind("<Button-5>", lambda e, s=side: zoom(e, s))
            canvas.bind("<ButtonPress-1>", lambda e, s=side: drag_start(e, s))
            canvas.bind("<B1-Motion>", lambda e, s=side: drag_motion(e, s))

        # --- Next/Prev Navigation Buttons ---
        def go_prev():
            if state["idx"] > 0:
                state["idx"] -= 1
                show(state["idx"])
        def go_next():
            if state["idx"] < len(state["matches"]) - 1:
                state["idx"] += 1
                show(state["idx"])
        btn_prev.configure(command=go_prev)
        btn_next.configure(command=go_next)

        # --- Keyboard Arrow navigation support ---
        left_canvas.bind("<Left>", lambda e: go_prev())
        left_canvas.bind("<Right>", lambda e: go_next())
        right_canvas.bind("<Left>", lambda e: go_prev())
        right_canvas.bind("<Right>", lambda e: go_next())
        left_canvas.focus_set()

        # --- Thumbnails, click to switch images ---
        def on_thumb(idx):
            state["idx"] = idx
            show(idx)
        for child in thumb_bar.winfo_children():
            child.destroy()
        for i, base in enumerate(matches):
            thumb_img = Image.open(orig_by_base[base]).resize((50, 50))
            imgtk = ImageTk.PhotoImage(thumb_img)
            f = ctk.CTkFrame(thumb_bar, fg_color="#e2ffe3" if i == 0 else "#f6f6f6", border_width=2, corner_radius=7)
            lbl_img = ctk.CTkLabel(f, image=imgtk, text="", width=60, height=30)
            lbl_img.image = imgtk
            lbl_img.pack()
            fname = os.path.basename(orig_by_base[base])
            lbl_txt = ctk.CTkLabel(f, text=fname[:12] + "…" if len(fname) > 14 else fname, font=ctk.CTkFont(size=10), width=72, anchor="center")
            lbl_txt.pack()
            f.pack(side="left", padx=1, pady=2)
            f.bind("<Button-1>", lambda e, idx=i: on_thumb(idx))
            lbl_img.bind("<Button-1>", lambda e, idx=i: on_thumb(idx))
            lbl_txt.bind("<Button-1>", lambda e, idx=i: on_thumb(idx))
        show(0)

        # Resize images if window size changes
        def resize_handler(event):
            show(state["idx"], resize_only=True)
        left_canvas.bind("<Configure>", resize_handler)
        right_canvas.bind("<Configure>", resize_handler)



    # ----- Plotting function -----
    def _dashboard_generate_and_open_plots(self, csv_path):
        import pandas as pd
        import plotly.graph_objs as go
        import plotly.express as px
        import numpy as np
        import tempfile, webbrowser
        from scipy.cluster.hierarchy import linkage, leaves_list
    
        df = pd.read_csv(csv_path)
        html_blocks = []
        classes = [c for c in df.columns if c not in ("Image", "Group")]
        class_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Vivid
    
        # --- 1. Absolute cell counts: grouped bar ---
        if "Group" in df.columns:
            # Each group: bar for each class
            group_abs = df.groupby("Group")[classes].sum()
            fig1 = go.Figure()
            for i, group in enumerate(group_abs.index):
                fig1.add_trace(go.Bar(
                    x=classes,
                    y=group_abs.loc[group].values,
                    name=str(group),
                    marker_color=class_palette[i % len(class_palette)],
                    text=[str(int(v)) for v in group_abs.loc[group].values],
                    textposition="outside",
                    hovertemplate=f"Group: {group}<br>Class: %{{x}}<br>Count: %{{y}}<extra></extra>"
                ))
            fig1.update_layout(
                barmode="group", 
                title="Absolute Cell Counts by Class per Group",
                xaxis_title="Class",
                yaxis_title="Total Count",
                template="plotly_white"
            )
        else:
            # single group, as before
            abs_sum = df[classes].sum()
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=classes, y=abs_sum, marker_color=class_palette[:len(classes)],text=[str(int(v)) for v in abs_sum],textposition="outside"))
            fig1.update_layout(title="Absolute Cell Counts by Class", xaxis_title="Class", yaxis_title="Total Count", template="plotly_white")
        html_blocks.append(go.Figure.to_html(fig1, include_plotlyjs='cdn', full_html=False))

        # --- X. Relative abundance: grouped bar ---
        if "Group" in df.columns:
            # Sum counts for each class, in each group
            group_totals = df.groupby("Group")[classes].sum()
            group_total_counts = group_totals.sum(axis=1)  # total cell count in each group
        
            fig_rel = go.Figure()
            for i, group in enumerate(group_totals.index):
                rel_pct = (group_totals.loc[group] / group_total_counts[group] * 100).round(2)
                fig_rel.add_trace(go.Bar(
                    x=classes,
                    y=rel_pct,
                    name=str(group),
                    marker_color=class_palette[i % len(class_palette)],
                    text=[f"{v:.1f}%" for v in rel_pct],
                    textposition="outside",
                    hovertemplate=f"Group: {group}<br>Class: %{{x}}<br>Rel %: %{{y:.2f}}%<extra></extra>"
                ))
            fig_rel.update_layout(
                barmode="group",
                title="Relative Abundance (%) by Class per Group",
                xaxis_title="Class",
                yaxis_title="(%) Abundance",
                template="plotly_white"
            )
        else:
            # Single group: total class counts across all images
            total = df[classes].sum()
            total_cells = total.sum()
            rel_pct = (total / total_cells * 100).round(2)
        
            fig_rel = go.Figure(
                data=[go.Bar(
                    x=classes,
                    y=rel_pct,
                    text=[f"{v:.1f}%" for v in rel_pct],
                    textposition="outside",
                    marker_color=class_palette[:len(classes)]
                )]
            )
            fig_rel.update_layout(
                title="Relative Abundance (%) of Morphotypes",
                xaxis_title="Class",
                yaxis_title="(%) Abundance",
                template="plotly_white"
            )
        
        html_blocks.append(go.Figure.to_html(fig_rel, include_plotlyjs=False, full_html=False))
           
        # --- 2. Mean relative abundance: grouped bar ---
        rel_df = df[classes].div(df[classes].sum(axis=1).replace(0, 1), axis=0)
        if "Group" in df.columns:
            rel_mean = rel_df.copy()
            rel_mean["Group"] = df["Group"]
            rel_grp = rel_mean.groupby("Group").mean() * 100
            fig2 = go.Figure()
            for i, group in enumerate(rel_grp.index):
                fig2.add_trace(go.Bar(
                    x=classes,
                    y=rel_grp.loc[group].values,
                    name=str(group),
                    marker_color=class_palette[i % len(class_palette)],
                    text=[f"{v:.1f}%" for v in rel_grp.loc[group].values],
                    textposition="outside",
                    hovertemplate=f"Group: {group}<br>Class: %{{x}}<br>Mean %: %{{y:.2f}}%<extra></extra>"
                ))
            fig2.update_layout(
                barmode="group", 
                title="Mean Relative Abundance (%) by Class per Group",
                xaxis_title="Class",
                yaxis_title="Mean Relative Abundance (%)",
                template="plotly_white"
            )
        else:
            rel_mean = rel_df.mean() * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=classes, y=rel_mean, marker_color=class_palette[:len(classes)],text=[f"{v:.1f}%" for v in rel_mean],textposition="outside"))
            fig2.update_layout(title="Mean Relative Abundance (%) by Class", xaxis_title="Class", yaxis_title="Mean Relative Abundance (%)", template="plotly_white")
        html_blocks.append(go.Figure.to_html(fig2, include_plotlyjs=False, full_html=False))
    
        # --- 3. Normalized stacked bar plot (composition) ---
        if "Group" in df.columns:
            prop_df = df.groupby("Group")[classes].sum()
            prop_df = prop_df.div(prop_df.sum(axis=1), axis=0).fillna(0)
            x = prop_df.index
        else:
            prop_df = rel_df
            x = df["Image"]
        fig3 = go.Figure()
        for i, cls in enumerate(classes):
            fig3.add_trace(go.Bar(
                x=x, y=prop_df[cls]*100, name=cls,
                marker_color=class_palette[i % len(class_palette)],
                hovertemplate=f"%{{y:.2f}}% {cls}<extra></extra>"
            ))
        fig3.update_layout(
            barmode="stack", 
            title="Normalized Stacked Bar Plot (Class Composition)", 
            xaxis_title="Image/Group", 
            yaxis_title="%", 
            template="plotly_white"
        )
        html_blocks.append(go.Figure.to_html(fig3, include_plotlyjs=False, full_html=False))
    
        # --- 4. HEATMAPS ---
    
        colorscale1 = 'Viridis'
        colorscale2 = 'Plasma'
    
        # Single group heatmap
        if "Group" not in df.columns:
            rel_img = df[classes].div(df[classes].sum(axis=1).replace(0, 1), axis=0)
            col_labels = df["Image"].astype(str).tolist()
            if len(rel_img) > 1:
                Z_col = linkage(rel_img.values, method="ward", metric="euclidean")
                col_leaves = leaves_list(Z_col)
                clustered_col_labels = [col_labels[i] for i in col_leaves]
                clustered_rel_img = rel_img.iloc[col_leaves, :]
            else:
                clustered_col_labels = col_labels
                clustered_rel_img = rel_img
            fig4 = go.Figure(data=go.Heatmap(
                z=clustered_rel_img.values.T,  # shape: classes x images
                x=clustered_col_labels,
                y=clustered_rel_img.columns,
                colorscale=colorscale2, colorbar=dict(title="Rel. Abundance"), zmin=0, zmax=1
            ))
            fig4.update_layout(
                title="Heatmap of Morphotype Distribution per Image (Relative Abundance, Clustered)",
                xaxis_title="Image (clustered)", yaxis_title="Class", template="plotly_white"
            )
            html_blocks.append(go.Figure.to_html(fig4, include_plotlyjs=False, full_html=False))
    
        # Multi-group heatmaps
        if "Group" in df.columns:
            # GROUP summary heatmap
            grp_prop = df.groupby("Group")[classes].mean()
            grp_rel = grp_prop.div(grp_prop.sum(axis=1), axis=0).fillna(0)
            if grp_rel.shape[0] > 1:
                Z_col = linkage(grp_rel.values, method="ward", metric="euclidean")
                col_leaves = leaves_list(Z_col)
                clustered_grp_names = list(grp_rel.index[col_leaves])
                clustered_grp_rel = grp_rel.iloc[col_leaves, :]
            else:
                clustered_grp_names = list(grp_rel.index)
                clustered_grp_rel = grp_rel
            fig5 = go.Figure(data=go.Heatmap(
                z=clustered_grp_rel.values.T,
                x=clustered_grp_names,
                y=clustered_grp_rel.columns,
                colorscale=colorscale1, colorbar=dict(title="Rel. Abundance"), zmin=0, zmax=1
            ))
            fig5.update_layout(
                title="Heatmap of Group-wise Morphotype Distribution (Relative Abundance, Clustered)",
                xaxis_title="Group (clustered)", yaxis_title="Class", template="plotly_white"
            )
            html_blocks.append(go.Figure.to_html(fig5, include_plotlyjs=False, full_html=False))
    
            # PER-IMAGE heatmap
            rel_img = df[classes].div(df[classes].sum(axis=1).replace(0, 1), axis=0).fillna(0)
            col_labels = [f"{g}_{i}" for g, i in zip(df["Group"], df["Image"])]
            if len(rel_img) > 1:
                Z_col = linkage(rel_img.values, method="ward", metric="euclidean")
                col_leaves = leaves_list(Z_col)
                clustered_col_labels = [col_labels[i] for i in col_leaves]
                clustered_rel_img = rel_img.iloc[col_leaves, :]
            else:
                clustered_col_labels = col_labels
                clustered_rel_img = rel_img
            fig6 = go.Figure(data=go.Heatmap(
                z=clustered_rel_img.values.T,
                x=clustered_col_labels,
                y=clustered_rel_img.columns,
                colorscale=colorscale2, colorbar=dict(title="Rel. Abundance"), zmin=0, zmax=1
            ))
            fig6.update_layout(
                title="Heatmap of Morphotype Distribution per Image (Relative Abundance, Clustered)",
                xaxis_title="Image (clustered)", yaxis_title="Class", template="plotly_white"
            )
            html_blocks.append(go.Figure.to_html(fig6, include_plotlyjs=False, full_html=False))
    
        # Write HTML file with all blocks
        html = "<html><head><meta charset='utf-8'></head><body>" + "\n<hr>\n".join(html_blocks) + "</body></html>"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as tmp:
            tmp.write(html)
            html_path = tmp.name
        webbrowser.open("file://" + html_path)
    


# ----- Main app loop -----
if __name__ == "__main__":
    app = MycoVisionApp()
    app.mainloop()
