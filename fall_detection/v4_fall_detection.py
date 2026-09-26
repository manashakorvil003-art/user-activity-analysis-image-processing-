import cv2
import json
import time
import math
import numpy as np
import tensorflow as tf
from collections import deque
from picamera2 import Picamera2


# ============================================================
# SMARTCARE SENTINEL V4
# Movement Tracking + Temporal Fall Detection
# ============================================================

BASE_DIR = "/home/korvil/HUMAN FALL DETECTION"

MODEL_PATH = BASE_DIR + "/models/smartcare_action_model.keras"
CLASS_PATH = BASE_DIR + "/models/classes.json"

WIDTH = 640
HEIGHT = 480

MODEL_SIZE = 224

DETECTION_INTERVAL = 5
AI_INTERVAL = 5

HISTORY_LENGTH = 15

FALL_CONFIRMATION_FRAMES = 5

HOG_SCALE = 1.03
HOG_STRIDE = (8, 8)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

fall_index = class_names.index("fall")
normal_index = class_names.index("normal")

print("Classes:", class_names)
print("AI model ready.")


# ============================================================
# PERSON DETECTOR
# ============================================================

hog = cv2.HOGDescriptor()

hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)

print("Person detector ready.")


# ============================================================
# CAMERA
# ============================================================

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (WIDTH, HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print("Camera started.")
print("Press Q to quit.")


# ============================================================
# HISTORY
# ============================================================

trajectory = deque(
    maxlen=HISTORY_LENGTH
)

height_history = deque(
    maxlen=HISTORY_LENGTH
)

angle_history = deque(
    maxlen=HISTORY_LENGTH
)


# ============================================================
# VARIABLES
# ============================================================

frame_number = 0

last_box = None

last_fall_percent = 0.0
last_normal_percent = 100.0

last_action = "NORMAL"

fall_counter = 0

state = "SAFE"

previous_time = time.time()

fps = 0.0


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def enhance_image(frame):

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge(
        (l, a, b)
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# ============================================================
# BODY ANGLE
# ============================================================

def body_angle(width, height):

    if height <= 0:
        return 0

    ratio = width / float(height)

    return math.degrees(
        math.atan(ratio)
    )


# ============================================================
# TEXT
# ============================================================

def put_text(
    frame,
    message,
    position,
    scale=0.55,
    color=(255, 255, 255),
    thickness=2
):

    cv2.putText(
        frame,
        message,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    frame_number += 1

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    frame_rgb = picam2.capture_array()

    frame = cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # ENHANCEMENT
    # --------------------------------------------------------

    enhanced = enhance_image(frame)


    # ========================================================
    # PERSON DETECTION
    # ========================================================

    if frame_number % DETECTION_INTERVAL == 0:

        boxes, weights = hog.detectMultiScale(
            frame,
            winStride=HOG_STRIDE,
            padding=(8, 8),
            scale=HOG_SCALE
        )

        best_box = None
        best_score = -999


        for box, weight in zip(
            boxes,
            weights
        ):

            score = float(
                np.asarray(weight).reshape(-1)[0]
            )

            if score > best_score:

                best_score = score

                best_box = tuple(
                    int(v)
                    for v in box
                )


        if best_box is not None:

            last_box = best_box


    # ========================================================
    # PERSON TRACKING
    # ========================================================

    if last_box is not None:

        x, y, w, h = last_box

        x = max(
            0,
            min(x, WIDTH - 1)
        )

        y = max(
            0,
            min(y, HEIGHT - 1)
        )

        w = max(
            1,
            min(w, WIDTH - x)
        )

        h = max(
            1,
            min(h, HEIGHT - y)
        )


        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        center_x = x + w // 2
        center_y = y + h // 2

        trajectory.append(
            (center_x, center_y)
        )


        # ----------------------------------------------------
        # BODY DATA
        # ----------------------------------------------------

        height_history.append(h)

        angle = body_angle(
            w,
            h
        )

        angle_history.append(angle)


        # ----------------------------------------------------
        # VERTICAL DISPLACEMENT
        # ----------------------------------------------------

        vertical_movement = 0

        if len(trajectory) >= 2:

            old_y = trajectory[0][1]

            new_y = trajectory[-1][1]

            vertical_movement = (
                new_y - old_y
            )


        # ====================================================
        # DRAW MOVEMENT TRAIL
        # ====================================================

        if len(trajectory) >= 2:

            for i in range(
                1,
                len(trajectory)
            ):

                p1 = trajectory[i - 1]

                p2 = trajectory[i]

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (255, 0, 255),
                    2
                )


        # ----------------------------------------------------
        # CENTER POINT
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (center_x, center_y),
            6,
            (255, 0, 255),
            -1
        )


        # ====================================================
        # AI CLASSIFICATION
        # ====================================================

        if frame_number % AI_INTERVAL == 0:

            crop = enhanced[
                y:y + h,
                x:x + w
            ]

            if crop.size > 0:

                crop = cv2.resize(
                    crop,
                    (
                        MODEL_SIZE,
                        MODEL_SIZE
                    )
                )

                crop = crop.astype(
                    np.float32
                )

                crop = (
                    crop / 127.5
                ) - 1.0

                crop = np.expand_dims(
                    crop,
                    axis=0
                )


                prediction = model.predict(
                    crop,
                    verbose=0
                )[0]


                last_fall_percent = (
                    float(
                        prediction[fall_index]
                    ) * 100
                )


                last_normal_percent = (
                    float(
                        prediction[normal_index]
                    ) * 100
                )


                if (
                    last_fall_percent
                    >
                    last_normal_percent
                ):

                    last_action = "FALL"

                else:

                    last_action = "NORMAL"


        # ====================================================
        # FALL FEATURES
        # ====================================================

        horizontal_body = (
            w > h * 1.15
        )

        sudden_downward = (
            vertical_movement > 25
        )

        large_angle = (
            angle > 48
        )

        strong_ai_fall = (
            last_fall_percent > 65
        )


        # ====================================================
        # FALL EVENT
        # ====================================================

        geometry_warning = (
            horizontal_body
            and
            (
                sudden_downward
                or
                large_angle
            )
        )


        possible_fall = (
            strong_ai_fall
            or
            geometry_warning
        )


        # ====================================================
        # STATE MACHINE
        # ====================================================

        if possible_fall:

            fall_counter += 1

        else:

            fall_counter = max(
                0,
                fall_counter - 1
            )


        if fall_counter == 0:

            state = "SAFE"

        elif fall_counter < 3:

            state = "POSSIBLE FALL"

        elif fall_counter < FALL_CONFIRMATION_FRAMES:

            state = "CONFIRMING"

        else:

            state = "FALL DETECTED"


        # ====================================================
        # COLORS
        # ====================================================

        if state == "FALL DETECTED":

            box_color = (
                0,
                0,
                255
            )

        elif state == "CONFIRMING":

            box_color = (
                0,
                165,
                255
            )

        elif state == "POSSIBLE FALL":

            box_color = (
                0,
                255,
                255
            )

        else:

            box_color = (
                0,
                255,
                0
            )


        # ====================================================
        # BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            3
        )


        put_text(
            frame,
            "PERSON",
            (x, max(25, y - 8)),
            0.55,
            box_color,
            2
        )


        # ====================================================
        # INFORMATION
        # ====================================================

        put_text(
            frame,
            "SMARTCARE SENTINEL V4",
            (20, 35),
            0.65,
            (255, 255, 255),
            2
        )

        put_text(
            frame,
            f"Action: {last_action}",
            (20, 68)
        )

        put_text(
            frame,
            f"Fall: {last_fall_percent:.1f}%",
            (20, 96),
            0.55,
            (0, 0, 255),
            2
        )

        put_text(
            frame,
            f"Normal: {last_normal_percent:.1f}%",
            (20, 124),
            0.55,
            (0, 255, 0),
            2
        )

        put_text(
            frame,
            f"Body angle: {angle:.1f}",
            (20, 152)
        )

        put_text(
            frame,
            f"Vertical movement: "
            f"{vertical_movement:.1f}",
            (20, 180)
        )

        put_text(
            frame,
            f"Fall counter: "
            f"{fall_counter}/"
            f"{FALL_CONFIRMATION_FRAMES}",
            (20, 208)
        )


        # ====================================================
        # STATE
        # ====================================================

        put_text(
            frame,
            f"STATE: {state}",
            (20, 250),
            0.75,
            box_color,
            3
        )


    else:

        # ====================================================
        # NO PERSON
        # ====================================================

        trajectory.clear()

        height_history.clear()

        angle_history.clear()

        fall_counter = 0

        state = "SAFE"

        put_text(
            frame,
            "SMARTCARE SENTINEL V4",
            (20, 35),
            0.65
        )

        put_text(
            frame,
            "NO PERSON DETECTED",
            (20, 75),
            0.65,
            (0, 0, 255),
            2
        )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    instant_fps = (
        1.0 /
        max(
            current_time - previous_time,
            0.001
        )
    )

    previous_time = current_time

    fps = (
        fps * 0.9
        +
        instant_fps * 0.1
    )


    put_text(
        frame,
        f"FPS: {fps:.1f}",
        (520, 450),
        0.5
    )

    put_text(
        frame,
        "Q = Exit",
        (20, 450),
        0.5
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "SmartCare Sentinel - V4",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

picam2.stop()

cv2.destroyAllWindows()

print("SmartCare Sentinel V4 stopped.")
