import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

# Sabitler
DISPLAY_W, DISPLAY_H = 400, 300

# Renkler ve stil
dark_bg = "#d0d4e0"
card_bg = "#fdf7fa"
accent = "#e5a4c8"
text_color = "#2d2d3a"


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔵 Görüntü İşleme Arayüzü")
        self.root.geometry("1400x800")
        self.root.configure(bg=dark_bg)

        self.original_image = None
        self.original_image_size = (0, 0)
        self.processed_cv = None
        self.processed_py = None

        self.template_rect = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.selection_mode = False
        self.img_tk = None

        self.build_ui()

    def build_ui(self):
        # Üst butonlar
        top = tk.Frame(self.root, bg=dark_bg)
        top.pack(pady=10)

        tk.Button(top, text="📁 Resim Yükle", command=self.load_image,
                  bg=accent, fg="white", font=("Arial", 11, "bold"), relief="flat", padx=20, pady=8).grid(row=0,
                                                                                                          column=0,
                                                                                                          padx=10)
        tk.Button(top, text="⚙ OpenCV ile Uygula", command=lambda: self.apply_selected(cv=True),
                  bg=accent, fg="white", font=("Arial", 11, "bold"), relief="flat", padx=20, pady=8).grid(row=0,
                                                                                                          column=1,
                                                                                                          padx=10)
        tk.Button(top, text="⚙ Python Algoritması ile Uygula", command=lambda: self.apply_selected(cv=False),
                  bg=accent, fg="white", font=("Arial", 11, "bold"), relief="flat", padx=20, pady=8).grid(row=0,
                                                                                                          column=2,
                                                                                                          padx=10)
        tk.Button(top, text="💾 Kaydet (OpenCV)", command=lambda: self.save_image(cv=True),
                  bg=accent, fg="white", font=("Arial", 11, "bold"), relief="flat", padx=20, pady=8).grid(row=0,
                                                                                                          column=3,
                                                                                                          padx=10)
        tk.Button(top, text="💾 Kaydet (Python)", command=lambda: self.save_image(cv=False),
                  bg=accent, fg="white", font=("Arial", 11, "bold"), relief="flat", padx=20, pady=8).grid(row=0,
                                                                                                          column=4,
                                                                                                          padx=10)

        self.message_label = tk.Label(top, text="İşlem hazır.", bg=dark_bg, fg="blue", font=("Arial", 10))
        self.message_label.grid(row=1, column=0, columnspan=5, pady=5)

        main_frame = tk.Frame(self.root, bg=dark_bg)
        main_frame.pack(fill="both", expand=True, pady=10)

        # Yan panel
        side_panel = tk.Frame(main_frame, bg=card_bg, width=260)
        side_panel.pack(side="left", fill="y", padx=10)
        tk.Label(side_panel, text="📌 İşlemler", bg=card_bg, fg=text_color, font=("Arial", 14, "bold")).pack(pady=10)

        canvas = tk.Canvas(side_panel, bg=card_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(side_panel, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=card_bg)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="top", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.operation_var = tk.StringVar(value="none")
        operations = [
            "Renkliyi Griye Dönüştür", "Parlaklık Ayarı", "Kontrast Ayarı", "Eşikleme",
            "Negatif Görüntü", "Alçak Geçiren Filtre",
            "Ortanca Filtre", "Laplace Filtre", "Sobel Yatay", "Sobel Dikey", "Prewitt",
            "Nesne Bulma (Şablon Eşleme)",
            "Ters Çevir", "Aynalama",
            "Erozyon", "Dilatasyon", "Açma", "Kapama"
        ]
        for op in operations:
            tk.Radiobutton(scroll_frame, text=op, variable=self.operation_var, value=op,
                           font=("Arial", 12), bg=card_bg, fg=text_color,
                           activebackground=card_bg, selectcolor=dark_bg, anchor="w", padx=10,
                           command=self.adjust_slider).pack(fill="x", pady=3)

        self.operation_var.trace_add("write", self.reset_template_selection)

        # --- YENİ: Değer Ayar Sürgüsü Paneli ---
        self.slider_frame = tk.LabelFrame(side_panel, text="İşlem Değeri", bg=card_bg, fg=text_color,
                                          font=("Arial", 10, "bold"))
        self.slider_frame.pack(side="bottom", fill="x", padx=10, pady=20)

        self.value_slider = tk.Scale(self.slider_frame, from_=0, to=255, orient="horizontal", bg=card_bg,
                                     font=("Arial", 10))
        self.value_slider.pack(fill="x", padx=5, pady=5)
        self.slider_frame.pack_forget()  # Başlangıçta gizli

        # Görüntü alanları
        right = tk.Frame(main_frame, bg=dark_bg)
        right.pack(side="left", expand=True, fill="both", padx=10)
        images_frame = tk.Frame(right, bg=dark_bg)
        images_frame.pack(expand=True, fill="both", pady=20)

        org_frame = tk.Frame(images_frame, bg=dark_bg)
        org_frame.pack(side="left", padx=10, expand=True)
        self.title_org = tk.Label(org_frame, text="Orijinal Resim", bg=dark_bg, fg=text_color,
                                  font=("Arial", 13, "bold"))
        self.title_org.pack(pady=5)
        self.label_org = tk.Canvas(org_frame, bg=card_bg, width=DISPLAY_W, height=DISPLAY_H, highlightthickness=1,
                                   highlightbackground="gray")
        self.label_org.pack(padx=10, pady=10)

        cv_frame = tk.Frame(images_frame, bg=dark_bg)
        cv_frame.pack(side="left", padx=10, expand=True)
        self.title_cv = tk.Label(cv_frame, text="OpenCV İşlem", bg=dark_bg, fg=text_color, font=("Arial", 13, "bold"))
        self.title_cv.pack(pady=5)
        self.label_cv = tk.Label(cv_frame, bg=card_bg)
        self.label_cv.pack(padx=10, pady=10)

        py_frame = tk.Frame(images_frame, bg=dark_bg)
        py_frame.pack(side="left", padx=10, expand=True)
        self.title_py = tk.Label(py_frame, text="Python Algoritması İşlem", bg=dark_bg, fg=text_color,
                                 font=("Arial", 13, "bold"))
        self.title_py.pack(pady=5)
        self.label_py = tk.Label(py_frame, bg=card_bg)
        self.label_py.pack(padx=10, pady=10)

    def adjust_slider(self):
        """Seçilen işleme göre sürgüyü gösterir ve ayarlar."""
        op = self.operation_var.get()
        if op == "Parlaklık Ayarı":
            self.slider_frame.pack(side="bottom", fill="x", padx=10, pady=20)
            self.value_slider.configure(from_=-100, to=100, resolution=1, label="Parlaklık Seviyesi")
            self.value_slider.set(40)
        elif op == "Kontrast Ayarı":
            self.slider_frame.pack(side="bottom", fill="x", padx=10, pady=20)
            self.value_slider.configure(from_=0.1, to=3.0, resolution=0.1, label="Kontrast Çarpanı")
            self.value_slider.set(1.5)
        elif op == "Eşikleme":
            self.slider_frame.pack(side="bottom", fill="x", padx=10, pady=20)
            self.value_slider.configure(from_=0, to=255, resolution=1, label="Eşik Değeri")
            self.value_slider.set(128)
        else:
            self.slider_frame.pack_forget()

    def reset_template_selection(self, *args):
        self.template_rect = None
        if self.img_tk:
            self.label_org.delete("all")
            self.label_org.create_image(0, 0, anchor="nw", image=self.img_tk)
        self.message_label.config(text="İşlem hazır.", fg="blue")
        self.label_org.unbind("<Button-1>")
        self.label_org.unbind("<B1-Motion>")
        self.label_org.unbind("<ButtonRelease-1>")
        self.selection_mode = False

    def enable_template_selection(self):
        self.message_label.config(text="Şablon seçmek için resim üzerine tıklayıp sürükleyin.", fg="red")
        self.selection_mode = True
        self.label_org.bind("<Button-1>", self.on_mouse_down)
        self.label_org.bind("<B1-Motion>", self.on_mouse_move)
        self.label_org.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_down(self, event):
        if not self.selection_mode: return
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id: self.label_org.delete(self.rect_id)
        self.rect_id = self.label_org.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                       outline="red", width=2)

    def on_mouse_move(self, event):
        if not self.selection_mode: return
        self.label_org.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if not self.selection_mode: return
        w_orig, h_orig = self.original_image_size
        scale_w, scale_h = w_orig / DISPLAY_W, h_orig / DISPLAY_H
        x1_orig = int(min(self.start_x, event.x) * scale_w)
        y1_orig = int(min(self.start_y, event.y) * scale_h)
        x2_orig = int(max(self.start_x, event.x) * scale_w)
        y2_orig = int(max(self.start_y, event.y) * scale_h)

        if x2_orig - x1_orig < 10 or y2_orig - y1_orig < 10:
            self.message_label.config(text="Hata: Alan çok küçük.", fg="red")
            return

        self.template_rect = (x1_orig, y1_orig, x2_orig, y2_orig)
        self.message_label.config(text="Şablon seçildi. Uygula'ya basın.", fg="green")
        self.selection_mode = False

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Resim Dosyaları", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not path: return
        self.original_image = cv2.imread(path)
        if self.original_image is None: return
        self.original_image_size = (self.original_image.shape[1], self.original_image.shape[0])
        self.reset_template_selection()
        self.show_image(self.original_image, self.label_org)

    def show_image(self, img, widget):
        if img is None: return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((DISPLAY_W, DISPLAY_H))
        img_tk = ImageTk.PhotoImage(img_pil)
        if isinstance(widget, tk.Canvas):
            self.img_tk = img_tk
            widget.delete("all")
            widget.create_image(0, 0, anchor="nw", image=self.img_tk)
            widget.image = self.img_tk
        else:
            widget.image = img_tk
            widget.configure(image=img_tk)

    def apply_selected(self, cv=True):
        if self.original_image is None: return
        op = self.operation_var.get()
        img = self.original_image.copy()

        if op == "Nesne Bulma (Şablon Eşleme)":
            if self.template_rect is None:
                self.enable_template_selection()
                return

        start_time = time.time()
        if cv:
            self.processed_cv = self.apply_cv(img, op)
            self.show_image(self.processed_cv, self.label_cv)
        else:
            self.processed_py = self.apply_python_manual(img, op)
            self.show_image(self.processed_py, self.label_py)

        self.message_label.config(text=f"İşlem bitti. Süre: {time.time() - start_time:.4f} s", fg="green")

    def apply_cv(self, img, op):
        val = self.value_slider.get()  # Sürgüdeki güncel değer
        if op == "Renkliyi Griye Dönüştür":
            return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif op == "Parlaklık Ayarı":
            return cv2.convertScaleAbs(img, alpha=1.0, beta=val)
        elif op == "Kontrast Ayarı":
            return cv2.convertScaleAbs(img, alpha=val, beta=0)
        elif op == "Eşikleme":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        elif op == "Negatif Görüntü":
            return 255 - img
        elif op == "Alçak Geçiren Filtre":
            return cv2.blur(img, (5, 5))
        elif op == "Ortanca Filtre":
            return cv2.medianBlur(img, 5)
        elif op == "Laplace Filtre":
            lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            return cv2.cvtColor(cv2.convertScaleAbs(lap), cv2.COLOR_GRAY2BGR)
        elif op == "Sobel Yatay":
            sob = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0)
            return cv2.cvtColor(cv2.convertScaleAbs(sob), cv2.COLOR_GRAY2BGR)
        elif op == "Sobel Dikey":
            sob = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1)
            return cv2.cvtColor(cv2.convertScaleAbs(sob), cv2.COLOR_GRAY2BGR)
        elif op == "Prewitt":
            kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            f = cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1, kernel)
            return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            # NESNE BULMA İŞLEMİ (OpenCV)
        elif op == "Nesne Bulma (Şablon Eşleme)":
            if self.template_rect is None:
                return img

            x1, y1, x2, y2 = self.template_rect
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            template = img_gray[y1:y2, x1:x2]
            w, h = template.shape[::-1]

            # Şablon eşleme uygula
            result_map = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            processed_img = img.copy()

            # --- YÖNTEM A: En Yüksek Skorlu Tek Eşleşme (Kırmızı) ---
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_map)

            if max_val >= 0.7:
                # En iyi eşleşmeyi bul ve Kırmızı (BGR: 0, 0, 255) ile çiz
                top_left_best = max_loc
                bottom_right_best = (top_left_best[0] + w, top_left_best[1] + h)
                cv2.rectangle(processed_img, top_left_best, bottom_right_best, (0, 0, 255), 3)

                # --- YÖNTEM B: Eşik Değerini Geçen Diğer Eşleşmeler (Mavi) ---
                threshold = 0.8
                loc = np.where(result_map >= threshold)

                for pt in zip(*loc[::-1]):
                    # Eğer bu nokta zaten en iyi eşleşme noktasıysa tekrar çizme (üst üste binmesin)
                    if abs(pt[0] - top_left_best[0]) < 5 and abs(pt[1] - top_left_best[1]) < 5:
                        continue

                bottom_right = (pt[0] + w, pt[1] + h)
                # Diğer eşleşmeleri Mavi (BGR: 255, 0, 0) ile çiz
                cv2.rectangle(processed_img, pt, bottom_right, (255, 0, 0), 2)

                self.message_label.config(
                    text=f"Eşleme bitti. En Yüksek Benzerlik: {max_val:.3f}. Yaklaşık {len(loc[0])} nokta bulundu.",
                    fg="green")
            else:
                self.message_label.config(
                    text=f"Uyarı: Eşleşme bulunamadı. En Yüksek: {max_val:.3f}",
                    fg="red")

            return processed_img

        elif op == "Ters Çevir":
            return cv2.rotate(img, cv2.ROTATE_180)
        elif op == "Aynalama":
            return cv2.flip(img, 1)
        elif op == "Erozyon":
            return cv2.erode(img, np.ones((3, 3), np.uint8))
        elif op == "Dilatasyon":
            return cv2.dilate(img, np.ones((3, 3), np.uint8))
        elif op == "Açma":
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        elif op == "Kapama":
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return img

    def apply_python_manual(self, img, op):
        h, w, c = img.shape
        result = img.astype(np.float32).copy()
        val = self.value_slider.get()
        def clamp(x):
            return max(0, min(255, int(x)))

        if op == "Renkliyi Griye Dönüştür":
            gray = np.mean(result, axis=2)
            result = np.stack([gray] * 3, axis=2)

        elif op == "Parlaklık Ayarı":
            result = np.clip(result + val, 0, 255)

        elif op == "Kontrast Ayarı":
            result = np.clip(result * val, 0, 255)

        elif op == "Eşikleme":
            gray = np.mean(result, axis=2)
            th = np.zeros_like(gray)
            for i in range(h):
                for j in range(w):
                    th[i, j] = 255 if gray[i, j] > val else 0
            result = np.stack([th] * 3, axis=2)

        elif op == "Negatif Görüntü":
            result = 255 - result

        elif op == "Alçak Geçiren Filtre":
            kernel = np.ones((3, 3)) / 9
            pad = 1
            padded = np.pad(result, ((pad, pad), (pad, pad), (0, 0)), 'edge')
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        result[i, j, k] = np.sum(padded[i:i + 3, j:j + 3, k] * kernel)

        elif op == "Ortanca Filtre":
            pad = 2
            padded = np.pad(result, ((pad, pad), (pad, pad), (0, 0)), 'edge')
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        # 5x5 pencerede median hesaplanır
                        result[i, j, k] = np.median(padded[i:i + 5, j:j + 5, k])

        elif op == "Laplace Filtre":
            gray = np.mean(result, axis=2)
            lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            pad = 1
            padded = np.pad(gray, pad, 'edge')
            lap = np.zeros_like(gray)
            for i in range(h):
                for j in range(w):
                    lap[i, j] = np.sum(padded[i:i + 3, j:j + 3] * lap_kernel)
            lap = np.clip(lap, 0, 255)
            result = np.stack([lap] * 3, axis=2)

        elif op == "Sobel Yatay":
            gray = np.mean(result, axis=2)
            kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            pad = 1
            padded = np.pad(gray, pad, 'edge')
            sobel_x = np.zeros_like(gray)
            for i in range(h):
                for j in range(w):
                    sobel_x[i, j] = np.sum(padded[i:i + 3, j:j + 3] * kx)
            sobel_x = np.clip(sobel_x, 0, 255)
            result = np.stack([sobel_x] * 3, axis=2)

        elif op == "Sobel Dikey":
            gray = np.mean(result, axis=2)
            ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            pad = 1
            padded = np.pad(gray, pad, 'edge')
            sobel_y = np.zeros_like(gray)
            for i in range(h):
                for j in range(w):
                    sobel_y[i, j] = np.sum(padded[i:i + 3, j:j + 3] * ky)
            sobel_y = np.clip(sobel_y, 0, 255)
            result = np.stack([sobel_y] * 3, axis=2)

        elif op == "Prewitt":
            gray = np.mean(result, axis=2)
            px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            pad = 1
            padded = np.pad(gray, pad, 'edge')
            prew = np.zeros_like(gray)
            for i in range(h):
                for j in range(w):
                    prew[i, j] = np.sum(padded[i:i + 3, j:j + 3] * px)
            prew = np.clip(prew, 0, 255)
            result = np.stack([prew] * 3, axis=2)


        elif op == "Nesne Bulma (Şablon Eşleme)":
            if self.template_rect is None:
                messagebox.showwarning("Uyarı", "Lütfen önce OpenCV butonu ile şablonu seçin.")
                return img.copy()

            # Şablon Eşleme (Template Matching), Python ile döngülerle elle yapıldığında performansı çok düşüktür.
            # Konsepti korumak için bilgilendirme mesajı ve bir kenar detayı filtresi (Sobel magnitude) gösterilir.
            messagebox.showinfo("Bilgilendirme",
                                "Şablon Eşleme (Template Matching) algoritması NumPy ile manuel olarak uygulandığında işlemci süresi (saatler) açısından verimli değildir. Bu nedenle sadece OpenCV versiyonunu kullanmanız önerilir. Bu kısım, yer tutucu olarak kenar büyülüğünü gösterir.")

            # Yer tutucu: Kenar Büyüklüğü filtresi döndürülür
            result = self.apply_python_manual(img, "Özellik Çıkarma (Kenar Büyüklüğü)")
            return result

        elif op == "Ters Çevir":
            result = result[::-1, ::-1, :]

        elif op == "Aynalama":
            result = result[:, ::-1, :]

        elif op == "Yakınlaştırma":
            zoom_factor = 1.5
            h_new = int(h * zoom_factor)
            w_new = int(w * zoom_factor)
            zoomed = np.zeros((h_new, w_new, c))
            for i in range(h_new):
                for j in range(w_new):
                    zoomed[i, j] = result[int(i / zoom_factor), int(j / zoom_factor)]
            result = zoomed

        elif op == "Uzaklaştırma":
            zoom_factor = 0.7
            h_new = int(h * zoom_factor)
            w_new = int(w * zoom_factor)
            small = np.zeros((h_new, w_new, c))
            for i in range(h_new):
                for j in range(w_new):
                    small[i, j] = result[int(i / zoom_factor), int(j / zoom_factor)]
            result = small

        elif op == "Erozyon":
            pad = 1
            padded = np.pad(result, ((pad, pad), (pad, pad), (0, 0)), 'edge')
            erode = np.zeros_like(result)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        erode[i, j, k] = np.min(padded[i:i + 3, j:j + 3, k])
            result = erode

        elif op == "Dilatasyon":
            pad = 1
            padded = np.pad(result, ((pad, pad), (pad, pad), (0, 0)), 'edge')
            dil = np.zeros_like(result)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        dil[i, j, k] = np.max(padded[i:i + 3, j:j + 3, k])
            result = dil

        elif op == "Açma":
            result = self.apply_python_manual(result, "Erozyon")
            result = self.apply_python_manual(result, "Dilatasyon")

        elif op == "Kapama":
            result = self.apply_python_manual(result, "Dilatasyon")
            result = self.apply_python_manual(result, "Erozyon")

        return np.uint8(result)

    # ---------------- Resim Kaydetme ----------------
    def save_image(self, cv=True):
        if cv:
            img_to_save = self.processed_cv
            if img_to_save is None:
                messagebox.showwarning("Uyarı", "OpenCV ile işlenmiş bir resim yok.")
                return
        else:
            img_to_save = self.processed_py
            if img_to_save is None:
                messagebox.showwarning("Uyarı", "Python algoritması ile işlenmiş bir resim yok.")
                return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG dosyaları", "*.png"), ("JPEG dosyaları", "*.jpg"), ("Tüm dosyalar", "*.*")]
        )
        if path:
            try:
                cv2.imwrite(path, img_to_save)
                messagebox.showinfo("Başarılı", f"Resim başarıyla kaydedildi:\n{path}")
            except Exception as e:
                messagebox.showerror("Hata", f"Kaydetme başarısız oldu:\n{e}")



if __name__ == '__main__':
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()