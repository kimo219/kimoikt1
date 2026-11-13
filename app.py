"""Washer gallery + simple local Face ID (educational LBPH example)."""
import os
import math
import time
import cv2
import numpy as np

# Always find washer photos next to this .py file, even when launched elsewhere.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

PHOTO_FILES = [f"washer{i}.jpg" if i > 1 else "washer.jpg" for i in range(1, 7)]
THUMB_W, THUMB_H = 240, 200
LABEL_H, COLS, PAD = 30, 4, 10
CELL_W = THUMB_W + PAD
CELL_H = THUMB_H + LABEL_H + PAD
GALLERY_WIN = "Gallery (click a photo, ESC - exit)"
RESULT_WIN = "Washer Quality Control"
FACE_FILE = os.path.join(BASE_DIR, "owner_faces.npz")


def face_id_login():
    """First run: press R to register. Later runs: verify this face locally."""
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        print("[ERROR] Face ID needs opencv-contrib-python, not opencv-python.")
        print("Run: python -m pip uninstall -y opencv-python opencv-contrib-python")
        print("Then: python -m pip install opencv-contrib-python numpy")
        return False

    detector = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    if detector.empty():
        print("[ERROR] Face detector XML could not be loaded.")
        return False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera unavailable: check webcam permission or close other applications.")
        cap.release()
        return False

    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(85, 85))
        if len(faces) != 1:
            return None, None
        x, y, w, h = faces[0]
        crop = cv2.equalizeHist(cv2.resize(gray[y:y+h, x:x+w], (120, 120)))
        return crop, (x, y, w, h)

    title = "Face ID (ESC to exit)"
    try:
        cv2.namedWindow(title)
        if not os.path.isfile(FACE_FILE):
            print("[FACE ID] First run: press R in the camera window to register your face.")
            print("[FACE ID] Photos are stored only in owner_faces.npz on your computer.")
            samples, capturing, frame_count = [], False, 0
            while len(samples) < 12:
                ok, frame = cap.read()
                if not ok:
                    print("[ERROR] Camera frame read failed.")
                    return False
                face, box = detect(frame)
                if box is not None:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                frame_count += 1
                if capturing and face is not None and frame_count % 6 == 0:
                    samples.append(face)
                label = (f"Register {len(samples)}/12" if capturing else "Press R to register")
                cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)
                cv2.imshow(title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    return False
                if key in (ord('r'), ord('R')):
                    capturing = True
            np.savez_compressed(FACE_FILE, faces=np.asarray(samples, dtype=np.uint8))
            print("[FACE ID] Registration complete. Local file:", FACE_FILE)
            return True

        with np.load(FACE_FILE, allow_pickle=False) as saved:
            samples = saved["faces"]
        if samples.ndim != 3 or samples.shape[1:] != (120, 120) or len(samples) < 2:
            print("[ERROR] Face data invalid. Delete owner_faces.npz to register again.")
            return False

        model = cv2.face.LBPHFaceRecognizer_create()
        model.train([im for im in samples], np.zeros(len(samples), np.int32))
        streak, begin = 0, time.monotonic()
        while time.monotonic() - begin < 30:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Camera frame read failed.")
                return False
            face, box = detect(frame)
            if face is not None:
                _, distance = model.predict(face)
                streak = streak + 1 if distance < 70 else 0
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 255, 0) if streak else (0, 0, 255), 2)
            else:
                streak = 0
            cv2.putText(frame, f"Face ID: {streak}/3", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(title, frame)
            if streak >= 3:
                print("[FACE ID] Authorized. Opening gallery.")
                return True
            if cv2.waitKey(1) & 0xFF == 27:
                return False
        print("[FACE ID] Authorization failed after 30 seconds.")
        return False
    except (cv2.error, OSError, KeyError, ValueError) as e:
        print("[FACE ID] Error:", e)
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()


def detect_crack(img_colored):
    """Ищет трещину, красит её красным. Возвращает (картинка, найдена_ли)."""
    draw_img = img_colored.copy()

    h, w = img_colored.shape[:2]
    center = (w // 2, h // 2)
    max_radius = min(center[0], center[1])

    # 1. Полярная развертка
    polar_w = w * 2
    polar_img = cv2.warpPolar(img_colored, (polar_w, h), center, max_radius,
                              cv2.WARP_POLAR_LINEAR)

    # 2. Подготовка
    gray_polar = cv2.cvtColor(polar_img, cv2.COLOR_BGR2GRAY)
    blurred_polar = cv2.GaussianBlur(gray_polar, (5, 5), 0)

    # 3. Фильтр вертикальных структур
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    blackhat_polar = cv2.morphologyEx(blurred_polar, cv2.MORPH_BLACKHAT, kernel_v)

    # 4. Порог
    _, thresh_polar = cv2.threshold(blackhat_polar, 55, 255, cv2.THRESH_BINARY)

    # 5. Склейка частей трещины
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    thresh_polar = cv2.morphologyEx(thresh_polar, cv2.MORPH_CLOSE, close_kernel)

    # 6. Контуры и фильтрация шума
    contours, _ = cv2.findContours(thresh_polar, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polar_mask = np.zeros_like(thresh_polar)
    real_crack_detected = False

    for cnt in contours:
        _, _, _, h_c = cv2.boundingRect(cnt)
        if h_c > 50 and cv2.contourArea(cnt) > 200:
            cv2.drawContours(polar_mask, [cnt], -1, 255, -1)
            real_crack_detected = True

    # 7. Обратная трансформация маски
    unpolar_mask = cv2.warpPolar(polar_mask, (w, h), center, max_radius,
                                 cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)

    # 8. Заливка красным
    crack_contours, _ = cv2.findContours(unpolar_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    smooth_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in crack_contours:
        cv2.drawContours(smooth_mask, [cnt], -1, 255, -1)

    smooth_mask = cv2.GaussianBlur(smooth_mask, (5, 5), 0)
    _, final_mask = cv2.threshold(smooth_mask, 100, 255, cv2.THRESH_BINARY)

    red_layer = np.full_like(draw_img, (0, 0, 255))
    mask_bool = final_mask.astype(bool)
    draw_img[mask_bool] = red_layer[mask_bool]

    return draw_img, real_crack_detected


# ====== Кэш результатов ======
results = {}


def process_file(file_name):
    """Обрабатывает фото (один раз, потом берёт из кэша)."""
    if file_name in results:
        return results[file_name]

    if not os.path.exists(file_name):
        print(f"[ОШИБКА] Файл '{file_name}' не найден!")
        return None

    img = cv2.imread(file_name)
    if img is None:
        print(f"[ОШИБКА] Не удалось прочитать '{file_name}'")
        return None

    draw_img, found = detect_crack(img)

    status_text = "BRAK (Trechina)" if found else "GODEN"
    status_color = (0, 0, 255) if found else (0, 255, 0)
    cv2.putText(draw_img, f"{file_name}: {status_text}", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)

    print(f"Файл: {file_name} -> {status_text}")
    results[file_name] = (draw_img, found)
    return results[file_name]


def make_thumbnail(file_name):
    """Миниатюра фото фиксированного размера (с сохранением пропорций)."""
    cell = np.full((THUMB_H, THUMB_W, 3), 60, dtype=np.uint8)
    img = cv2.imread(file_name) if os.path.exists(file_name) else None

    if img is None:
        cv2.putText(cell, "NOT FOUND", (20, THUMB_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return cell

    h, w = img.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (THUMB_W - nw) // 2, (THUMB_H - nh) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = small
    return cell


thumbnails = [make_thumbnail(f) for f in PHOTO_FILES]


def draw_gallery(selected=None):
    rows = max(1, math.ceil(len(PHOTO_FILES) / COLS))
    canvas = np.full((rows * CELL_H + PAD, COLS * CELL_W + PAD, 3), 30, dtype=np.uint8)

    for i, file_name in enumerate(PHOTO_FILES):
        r, c = divmod(i, COLS)
        x = PAD + c * CELL_W
        y = PAD + r * CELL_H

        canvas[y:y + THUMB_H, x:x + THUMB_W] = thumbnails[i]

        # Рамка: красная = брак, зелёная = годен, белая = выбрано сейчас
        if file_name in results:
            found = results[file_name][1]
            color = (0, 0, 255) if found else (0, 200, 0)
            cv2.rectangle(canvas, (x, y), (x + THUMB_W - 1, y + THUMB_H - 1), color, 3)
        if file_name == selected:
            cv2.rectangle(canvas, (x - 2, y - 2),
                          (x + THUMB_W + 1, y + THUMB_H + 1), (255, 255, 255), 2)

        cv2.putText(canvas, file_name, (x + 4, y + THUMB_H + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return canvas


selected_file = None


def on_mouse(event, x, y, flags, param):
    global selected_file
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    c = (x - PAD) // CELL_W
    r = (y - PAD) // CELL_H
    if c < 0 or c >= COLS or r < 0:
        return

    in_x = (x - PAD) % CELL_W < THUMB_W
    in_y = (y - PAD) % CELL_H < THUMB_H + LABEL_H
    if not (in_x and in_y):
        return

    idx = r * COLS + c
    if idx >= len(PHOTO_FILES):
        return

    file_name = PHOTO_FILES[idx]
    selected_file = file_name

    res = process_file(file_name)
    if res is None:
        return

    draw_img = res[0]
    h = draw_img.shape[0]
    display_img = cv2.resize(draw_img, None, fx=0.7, fy=0.7) if h > 800 else draw_img
    cv2.imshow(RESULT_WIN, display_img)


def main():
    # Авторизация перед открытием галереи.
    if not face_id_login():
        return

    cv2.namedWindow(GALLERY_WIN)
    cv2.setMouseCallback(GALLERY_WIN, on_mouse)

    while True:
        cv2.imshow(GALLERY_WIN, draw_gallery(selected_file))
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):
            break
        if cv2.getWindowProperty(GALLERY_WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
