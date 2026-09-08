import os

# ============================================================
# FFmpeg / OPENCV
# ============================================================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import time
import threading
import re
import requests

from datetime import datetime

from ultralytics import YOLO
import easyocr
from openpyxl import Workbook, load_workbook


# ============================================================
# CAMERA
# ============================================================

RTSP_URL = "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/media/video1"


# ============================================================
# MODEL
# ============================================================

VEHICLE_MODEL = "yolo11n.pt"


# ============================================================
# SETTINGS
# ============================================================

VEHICLE_CONFIDENCE = 0.30
AI_SIZE = 416


# ============================================================
# VIRTUAL LINE
# ============================================================

LINE_POSITION = 0.50
LINE_TOLERANCE = 8


# ============================================================
# ANPR AREA
# ============================================================

ANPR_X1 = 0.20
ANPR_Y1 = 0.20
ANPR_X2 = 0.80
ANPR_Y2 = 0.48


# ============================================================
# OCR
# ============================================================

OCR_LANGUAGES = ["en"]
OCR_COOLDOWN = 0.15


# ============================================================
# VEHICLE COOLDOWN
# ============================================================

BARRIER_COOLDOWN = 10


# ============================================================
# EXCEL
# ============================================================

EXCEL_FILE = "authorized_plates.xlsx"
EXCEL_COLUMN = 1


# ============================================================
# BARRIER
# ============================================================

BARRIER_OPEN_URL = "http://192.168.1.100/open"


# ============================================================
# GLOBAL
# ============================================================

latest_frame = None
frame_lock = threading.Lock()
running = True


# ============================================================
# VEHICLE TRACKING
# ============================================================

vehicle_sides = {}
vehicle_last_ocr = {}
vehicle_last_open = {}
vehicle_plate = {}
vehicle_plate_confidence = {}


# ============================================================
# SCREEN MESSAGE
# ============================================================

success_message = ""
success_plate = ""
success_time = 0
SUCCESS_DISPLAY_TIME = 5.0


# ============================================================
# ERROR MESSAGE
# ============================================================

error_message = ""
error_plate = ""
error_time = 0
ERROR_DISPLAY_TIME = 5.0


# ============================================================
# CURRENT PLATE
# ============================================================

current_plate = ""
current_plate_time = 0
CURRENT_PLATE_DISPLAY_TIME = 3


# ============================================================
# OCR GLOBAL COOLDOWN
# ============================================================

last_global_ocr = 0
is_ocr_processing = False


# ============================================================
# COUNTERS
# ============================================================

vehicle_count = 0


# ============================================================
# REPORT
# ============================================================

REPORT_FOLDER = "report"
os.makedirs(REPORT_FOLDER, exist_ok=True)

current_date = datetime.now().strftime("%Y-%m-%d")
report_file = os.path.join(REPORT_FOLDER, current_date + "_anpr.xlsx")


# ============================================================
# CREATE REPORT
# ============================================================

def create_report_file():
    if not os.path.exists(report_file):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "ANPR"
        sheet.append([
            "Date",
            "Time",
            "Vehicle_ID",
            "Plate",
            "Authorized",
            "Barrier"
        ])
        workbook.save(report_file)
        workbook.close()

create_report_file()


# ============================================================
# NORMALIZE PLATE
# ============================================================

def normalize_plate(text):
    if text is None:
        return ""

    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    if len(text) == 7:
        chars = list(text)

        digit_map = {
            "O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8"
        }

        for i in [0, 1, 4, 5, 6]:
            chars[i] = digit_map.get(chars[i], chars[i])

        letter_map = {
            "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"
        }

        for i in [2, 3]:
            chars[i] = letter_map.get(chars[i], chars[i])

        text = "".join(chars)

    return text


# ============================================================
# LOAD EXCEL
# ============================================================

def load_authorized_plates():
    plates = set()

    if not os.path.exists(EXCEL_FILE):
        print("\nWARNING:\n" + f"{EXCEL_FILE} tapılmadı!\n")
        return plates

    try:
        workbook = load_workbook(EXCEL_FILE, read_only=True)
        sheet = workbook.active

        for row in sheet.iter_rows(min_col=EXCEL_COLUMN, max_col=EXCEL_COLUMN):
            value = row[0].value
            if value is None:
                continue

            plate = normalize_plate(str(value))

            if len(plate) < 5:
                continue

            if plate:
                plates.add(plate)

        workbook.close()

    except Exception as e:
        print("Excel error:", e)

    print(f"Excel-dən {len(plates)} nömrə yükləndi.")

    for plate in sorted(plates):
        print("  ", plate)

    return plates


authorized_plates = load_authorized_plates()


def reload_excel():
    global authorized_plates
    authorized_plates = load_authorized_plates()


# ============================================================
# SAVE EVENT
# ============================================================

def save_event(vehicle_id, plate, authorized, barrier):
    global current_date
    global report_file

    today = datetime.now().strftime("%Y-%m-%d")

    if today != current_date:
        current_date = today
        report_file = os.path.join(REPORT_FOLDER, current_date + "_anpr.xlsx")
        create_report_file()

    now = datetime.now()

    try:
        workbook = load_workbook(report_file)
        sheet = workbook.active

        sheet.append([
            today,
            now.strftime("%H:%M:%S"),
            vehicle_id,
            plate,
            "YES" if authorized else "NO",
            barrier
        ])

        workbook.save(report_file)
        workbook.close()

    except Exception as e:
        print("Report error:", e)


# ============================================================
# BARRIER OPEN
# ============================================================

def open_barrier_async():
    try:
        response = requests.get(BARRIER_OPEN_URL, timeout=2)
        print("Barrier response:", response.status_code)
    except Exception as e:
        print("Barrier error:", e)


def open_barrier():
    print("\n>>> BARRIER OPEN COMMAND")
    threading.Thread(target=open_barrier_async, daemon=True).start()
    return True


# ============================================================
# OCR PROCESSİNG
# ============================================================

def read_plate(plate_crop, reader):
    if plate_crop is None or plate_crop.size == 0:
        return "", 0

    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.medianBlur(enhanced, 3)

        resized = cv2.resize(
            blurred,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_LINEAR
        )

        results = reader.readtext(resized, detail=1)

        if not results:
            return "", 0

        best_text = ""
        best_conf = 0

        for item in results:
            text = item[1]
            confidence = float(item[2])

            normalized = normalize_plate(text)

            if len(normalized) >= 5:
                if confidence > best_conf:
                    best_text = normalized
                    best_conf = confidence

        return (best_text, best_conf)

    except Exception as e:
        print("OCR error:", e)
        return "", 0


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_thread():
    global latest_frame
    global running

    cap = None

    while running:
        if cap is None or not cap.isOpened():
            print("Connecting to Uniview camera...")

            if cap is not None:
                cap.release()

            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print("Camera connection failed.")
                time.sleep(1)
                continue

            print("Camera connected.")

        ret, frame = cap.read()

        if not ret:
            print("Camera frame lost.")
            cap.release()
            cap = None
            time.sleep(0.5)
            continue

        height, width = frame.shape[:2]

        if width > 1280:
            scale = 1280 / width
            frame = cv2.resize(
                frame,
                (1280, int(height * scale)),
                interpolation=cv2.INTER_AREA
            )

        with frame_lock:
            latest_frame = frame

        time.sleep(0.001)

    if cap is not None:
        cap.release()


# ============================================================
# ASYNC OCR RUNNER (EKRANIN DONDURULMASININ QARŞISINI ALIR)
# ============================================================

def process_ocr_async(plate_crop, reader):
    global is_ocr_processing
    global current_plate
    global current_plate_time
    global success_message
    global success_plate
    global success_time
    global error_message
    global error_plate
    global error_time
    global vehicle_count

    try:
        plate_text, confidence = read_plate(plate_crop, reader)

        if plate_text:
            now = time.time()
            print("\n================================")
            print("ANPR OCR")
            print(f"OCR: {plate_text}")
            print(f"OCR confidence: {confidence:.2f}")

            current_plate = plate_text
            current_plate_time = now

            authorized = (plate_text in authorized_plates)

            if authorized:
                print("\n################################")
                print("SUCCESSFUL")
                print(f"PLATE: {plate_text}")
                print("AUTHORIZED: YES")

                open_barrier()

                success_message = "SUCCESSFUL"
                success_plate = str(plate_text)
                success_time = time.time()

                save_event(0, plate_text, True, "OPENED")
                vehicle_count += 1

            else:
                print("\nNOT AUTHORIZED")
                print(f"PLATE: {plate_text}")

                error_message = "NOT AUTHORIZED"
                error_plate = str(plate_text)
                error_time = time.time()

                save_event(0, plate_text, False, "NO OPEN")

    finally:
        is_ocr_processing = False


# ============================================================
# AI THREAD
# ============================================================

def ai_thread():
    global running
    global last_global_ocr
    global is_ocr_processing

    print("Loading vehicle model...")
    vehicle_model = YOLO(VEHICLE_MODEL)

    print("Loading OCR...")
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)

    print("ANPR system ready.")

    while running:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue

            frame = latest_frame.copy()

        try:
            frame_height, frame_width = frame.shape[:2]

            # Vehicle tracking
            results = vehicle_model.track(
                frame,
                persist=True,
                classes=[2, 3, 5, 7],
                conf=VEHICLE_CONFIDENCE,
                imgsz=AI_SIZE,
                tracker="bytetrack.yaml",
                verbose=False
            )

            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)

                line_y = int(frame_height * LINE_POSITION)

                for box, vehicle_id in zip(boxes, ids):
                    vehicle_id = int(vehicle_id)
                    x1, y1, x2, y2 = map(int, box)

                    center_y = int((y1 + y2) / 2)

                    if center_y < (line_y - LINE_TOLERANCE):
                        current_side = "TOP"
                    elif center_y > (line_y + LINE_TOLERANCE):
                        current_side = "BOTTOM"
                    else:
                        current_side = "LINE"

                    vehicle_sides[vehicle_id] = current_side

            # ANİ VƏ GECİKMƏSİZ OXUMA
            now = time.time()
            if not is_ocr_processing and (now - last_global_ocr > OCR_COOLDOWN):
                last_global_ocr = now

                anpr_x1 = int(frame_width * ANPR_X1)
                anpr_y1 = int(frame_height * ANPR_Y1)
                anpr_x2 = int(frame_width * ANPR_X2)
                anpr_y2 = int(frame_height * ANPR_Y2)

                plate_crop = frame[
                    max(0, anpr_y1):min(frame_height, anpr_y2),
                    max(0, anpr_x1):min(frame_width, anpr_x2)
                ].copy()

                is_ocr_processing = True
                threading.Thread(
                    target=process_ocr_async,
                    args=(plate_crop, reader),
                    daemon=True
                ).start()

        except Exception as e:
            print("AI error:", e)

        time.sleep(0.01)


# ============================================================
# START THREADS
# ============================================================

camera = threading.Thread(target=camera_thread, daemon=True)
ai = threading.Thread(target=ai_thread, daemon=True)

camera.start()
ai.start()


print("\n==========================================")
print(" UNIVIEW ANPR BARRIER SYSTEM")
print("==========================================")
print("Press Q to exit.\n")


# ============================================================
# DISPLAY LOOP
# ============================================================

while True:
    with frame_lock:
        if latest_frame is None:
            time.sleep(0.005)
            continue

        frame = latest_frame.copy()

    height, width = frame.shape[:2]
    line_y = int(height * LINE_POSITION)

    anpr_x1 = int(width * ANPR_X1)
    anpr_y1 = int(height * ANPR_Y1)
    anpr_x2 = int(width * ANPR_X2)
    anpr_y2 = int(height * ANPR_Y2)

    # Trigger Line & ANPR Area
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
    cv2.putText(frame, "ANPR TRIGGER LINE", (20, line_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(frame, (anpr_x1, anpr_y1), (anpr_x2, anpr_y2), (0, 255, 255), 2)
    cv2.putText(frame, "ANPR AREA", (anpr_x1 + 10, anpr_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ========================================================
    # SUCCESSFUL OVERLAY
    # ========================================================
    current_now = time.time()

    if success_plate and (current_now - success_time < SUCCESS_DISPLAY_TIME):
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(frame, (int(width/2) - 125, 5), (int(width/2) + 125, 80), (0, 0, 0), -1)

        text1 = "SUCCESSFUL"
        size1 = cv2.getTextSize(text1, font, 0.6, 2)[0]
        x1 = int((width - size1[0]) / 2)
        cv2.putText(frame, text1, (x1, 25), font, 0.6, (0, 255, 0), 2)

        text2 = "AUTHORIZED"
        size2 = cv2.getTextSize(text2, font, 0.4, 1)[0]
        x2 = int((width - size2[0]) / 2)
        cv2.putText(frame, text2, (x2, 45), font, 0.4, (0, 255, 0), 1)

        text3 = f"PLATE: {success_plate}"
        size3 = cv2.getTextSize(text3, font, 0.5, 2)[0]
        x3 = int((width - size3[0]) / 2)
        cv2.putText(frame, text3, (x3, 70), font, 0.5, (0, 255, 0), 2)

    # ========================================================
    # ERROR OVERLAY
    # ========================================================
    elif error_plate and (current_now - error_time < ERROR_DISPLAY_TIME):
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(frame, (int(width/2) - 250, 10), (int(width/2) + 250, 130), (0, 0, 0), -1)

        text1 = "NOT AUTHORIZED"
        size1 = cv2.getTextSize(text1, font, 1.1, 3)[0]
        x1 = int((width - size1[0]) / 2)
        cv2.putText(frame, text1, (x1, 50), font, 1.1, (0, 0, 255), 3)

        text2 = f"PLATE: {error_plate}"
        size2 = cv2.getTextSize(text2, font, 0.9, 2)[0]
        x2 = int((width - size2[0]) / 2)
        cv2.putText(frame, text2, (x2, 100), font, 0.9, (0, 0, 255), 2)

    # Information panel
    cv2.rectangle(frame, (10, 10), (300, 115), (0, 0, 0), -1)
    cv2.putText(frame, f"AUTHORIZED: {len(authorized_plates)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"OPEN EVENTS: {vehicle_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("UNIVIEW ANPR BARRIER SYSTEM", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        running = False
        break
    if key == ord("r"):
        print("Reloading Excel...")
        reload_excel()

    time.sleep(0.005)

running = False
cv2.destroyAllWindows()
