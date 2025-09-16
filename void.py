import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class FaceApp:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Outer lip landmark indices
        self.outer_lip_upper = [61, 40, 37, 0, 267, 270, 291]
        self.outer_lip_lower = [61, 146, 91, 181, 84, 17, 314,
                                405, 321, 375, 291]

        self.original_image = None
        self.processing = False

    def get_landmarks(self, image):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        points = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks.landmark])
        return points

    def process_lips(self, image, factor):

        if abs(factor - 1.0) < 0.01:
            return image

        landmarks = self.get_landmarks(image)
        if landmarks is None:
            return image

        outer_upper = landmarks[self.outer_lip_upper]
        outer_lower = landmarks[self.outer_lip_lower]
        outer_lip = np.vstack([outer_upper, outer_lower])

        # Lip corners
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mid_x = (left_corner[0] + right_corner[0]) / 2
        mid_y = (left_corner[1] + right_corner[1]) / 2
        lip_width = right_corner[0] - left_corner[0]

        def adjust_points(points, direction):
            new_pts = points.copy()
            for i, point in enumerate(points):
                if np.array_equal(point, left_corner) or np.array_equal(point, right_corner):
                    continue
                rel_pos = 1.0 - abs((point[0] - mid_x) / (lip_width / 2.0))
                rel_pos = max(rel_pos, 0)
                dy = point[1] - mid_y
                new_y = point[1] + direction * abs(dy) * (factor - 1.0) * rel_pos
                new_pts[i] = [point[0], new_y]
            return new_pts

        new_upper = adjust_points(outer_upper, -1)
        new_lower = adjust_points(outer_lower, +1)
        new_outer = np.vstack([new_upper, new_lower])

        # Original lip mask
        lip_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(lip_mask, [outer_lip.astype(np.int32)], 255)

        # --- Estimate skin tone around original lips (initial guess) ---
        dilated = cv2.dilate(lip_mask, np.ones((25, 25), np.uint8), iterations=1)
        skin_ring = cv2.subtract(dilated, lip_mask)
        skin_pixels = image[skin_ring == 255]
        if len(skin_pixels) > 0:
            skin_color = np.median(skin_pixels, axis=0).astype(np.uint8)
        else:
            skin_color = (180, 130, 100)

        # Lip texture
        lip_texture = cv2.bitwise_and(image, image, mask=lip_mask)

        # Warp lips to new shape
        src_pts = outer_lip.astype(np.float32)
        dst_pts = new_outer.astype(np.float32)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return image
        warped = cv2.warpPerspective(lip_texture, M, (image.shape[1], image.shape[0]))

        # New expanded lip mask
        new_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(new_mask, [new_outer.astype(np.int32)], 255)

        # Fill gaps with SKIN tone - better approach
        new_lips = cv2.bitwise_and(warped, warped, mask=new_mask)

        # Detect black/void areas in the warped lips
        gray_warped = cv2.cvtColor(new_lips, cv2.COLOR_BGR2GRAY)
        void_mask = (gray_warped < 10).astype(np.uint8) * 255  # Very dark areas
        void_mask = cv2.bitwise_and(void_mask, new_mask)  # Only within lip area

        # Fill voids with skin color
        if np.sum(void_mask) > 0:
            skin_filler = np.full_like(image, skin_color, dtype=np.uint8)
            new_lips = np.where(void_mask[..., np.newaxis] > 0, skin_filler, new_lips)
        # --- Seamless (Poisson) blending to avoid any visible polygon/outline ---
        # Slightly erode the mask so blending happens inside the boundary
        mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blend_mask = cv2.erode(new_mask, mask_kernel, iterations=1)

        # Compute clone center from mask moments
        m = cv2.moments(blend_mask)
        if m["m00"] == 0:
            return image
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"]) 

        # Source object is the warped lips on black background
        src = new_lips

        def soft_blend_fallback(src_img, dst_img, mask_img):
            alpha = cv2.GaussianBlur(mask_img, (0, 0), 3.0).astype(np.float32) / 255.0
            alpha_3 = alpha[..., np.newaxis]
            out = alpha_3 * src_img.astype(np.float32) + (1.0 - alpha_3) * dst_img.astype(np.float32)
            return np.clip(out, 0, 255).astype(np.uint8)

        if not hasattr(cv2, 'seamlessClone'):
            return soft_blend_fallback(src, image, blend_mask)

        try:
            result = cv2.seamlessClone(src, image, blend_mask, (cx, cy), cv2.MIXED_CLONE)
            return result
        except Exception:
            try:
                result = cv2.seamlessClone(src, image, blend_mask, (cx, cy), cv2.NORMAL_CLONE)
                return result
            except Exception:
                return soft_blend_fallback(src, image, blend_mask)
       
    def update_display(self):
        if self.original_image is None or self.processing:
            return
        self.processing = True

        val = self.lip_scale.get()
        result = self.process_lips(self.original_image.copy(), val)

        h, w = result.shape[:2]
        display_h = 400
        display_w = int(w * display_h / h)
        resized = cv2.resize(result, (display_w, display_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.config(width=display_w, height=display_h)
        self.canvas.create_image(display_w//2, display_h//2, image=self.photo)
        self.lip_label.config(text=f"{val:.1f}")

        self.processing = False
        self.current_result = result

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Could not load image")
            return
        if self.get_landmarks(image) is None:
            messagebox.showwarning("Warning", "No face detected")
            return
        self.original_image = image
        self.current_result = image.copy()
        self.lip_scale.set(1.0)
        self.update_display()
        messagebox.showinfo("Success", "Image loaded successfully!")

    def save_image(self):
        if not hasattr(self, 'current_result'):
            messagebox.showwarning("Warning", "No image to save")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if file_path:
            cv2.imwrite(file_path, self.current_result)
            messagebox.showinfo("Success", "Image saved!")

    def reset(self):
        self.lip_scale.set(1.0)
        if self.original_image is not None:
            self.update_display()

    def run(self):
        root = tk.Tk()
        root.title("Lip Augmentation Tool")
        root.geometry("700x600")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Lip Augmentation Tool",
                  font=("Arial", 14, "bold")).pack(pady=(0, 10))

        self.canvas = tk.Canvas(main_frame, bg="lightgray", width=500, height=400)
        self.canvas.pack(pady=10)

        controls = ttk.Frame(main_frame)
        controls.pack(fill=tk.X, pady=10)

        lip_frame = ttk.Frame(controls)
        lip_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lip_frame, text="Lip Enlargement (0.5–2.0):").pack(side=tk.LEFT)
        self.lip_scale = tk.Scale(
            lip_frame, from_=0.5, to=2.0, resolution=0.1,
            orient=tk.HORIZONTAL, command=lambda x: self.update_display()
        )
        self.lip_scale.set(1.0)
        self.lip_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        self.lip_label = ttk.Label(lip_frame, text="1.0")
        self.lip_label.pack(side=tk.RIGHT)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)

        root.mainloop()


if __name__ == "__main__":
    print("Starting Lip Augmentation Tool...")
    app = FaceApp()
    app.run()