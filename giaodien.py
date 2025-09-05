import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import joblib
import numpy as np
from insightface.app import FaceAnalysis

# --- Load model SVM ---
save_path = "face_buffalo_l_svm.pkl"   # thay bằng đường dẫn pkl của bạn
svm_loaded = joblib.load(save_path)

# --- Khởi tạo InsightFace ---
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# --- Tkinter setup ---
root = tk.Tk()
root.title("Face Recognition Demo")
root.geometry("900x700")


cap = None
panel = None

id_to_name = {
    1: "Minh Anh",
    2: "Linh",
    3: "Vũ",
    4: "Long"
}


def recognize_and_show(img_bgr):
    """Nhận diện khuôn mặt và gắn nhãn"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)

    if len(faces) == 0:
        cv2.putText(img_rgb, "Khong tim thay khuon mat", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        for face in faces:
            emb_test = face.embedding.reshape(1, -1)
            pred_label = svm_loaded.predict(emb_test)[0]
            pred_prob = svm_loaded.predict_proba(emb_test).max()

            name = id_to_name.get(pred_label, f"ID {pred_label}")
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"Ten : {name} ({pred_prob:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    return img_rgb


def open_file():
    global panel, cap
    if cap is not None:
        cap.release()
        cap = None
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not file_path:
        return
    img_bgr = cv2.imread(file_path)
    img_rgb = recognize_and_show(img_bgr)
    img_rgb = cv2.resize(img_rgb, (640, 480))
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    panel.config(image=img_tk)
    panel.image = img_tk


def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    show_frame()


def show_frame():
    global cap, panel
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_rgb = recognize_and_show(frame)
            img_rgb = cv2.resize(img_rgb, (640, 480))
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            panel.config(image=img_tk)
            panel.image = img_tk
        panel.after(10, show_frame)

# Khung hiển thị
panel = tk.Label(root, bg="gray")
panel.pack(pady=20)

# Nút chức năng
btn_frame = tk.Frame(root)
btn_frame.pack()


btn_file = tk.Button(btn_frame, text="Chọn file", command=open_file, width=15, height=2, bg="#FF9800", fg="white")
btn_file.pack(side="left", padx=10)


btn_cam = tk.Button(btn_frame, text="Mở camera", command=open_camera, width=15, height=2, bg="#4CAF50", fg="white")
btn_cam.pack(side="left", padx=10)


root.mainloop()
