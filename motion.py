import cv2
import numpy as np
import pyautogui
from collections import deque
import time

CAPTURE_DEVICE = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SKIN_HSV_LOWER = np.array([0, 30, 60])
SKIN_HSV_UPPER = np.array([20, 150, 255])

MIN_CONTOUR_AREA = 2000
POSITION_HISTORY = 6
VEL_SMOOTH = 0.6

H_DEADZONE_RATIO = 0.12
H_MOVE_COOLDOWN = 0.12

JUMP_VEL_THRESHOLD = -450.0
SLIDE_DOUBLE_TIME = 0.45
JUMP_COOLDOWN = 0.35
SLIDE_COOLDOWN = 0.6

KEY_LEFT = 'left'
KEY_RIGHT = 'right'
KEY_JUMP = 'space'
KEY_SLIDE = 'down'

SHOW_DEBUG = True

class SmoothedVelocity:
    def __init__(self, alpha=VEL_SMOOTH):
        self.alpha = alpha
        self.vx = 0.0
        self.vy = 0.0
        self.initialized = False

    def update(self, raw_vx, raw_vy):
        if not self.initialized:
            self.vx = raw_vx
            self.vy = raw_vy
            self.initialized = True
        else:
            self.vx = self.alpha * raw_vx + (1 - self.alpha) * self.vx
            self.vy = self.alpha * raw_vy + (1 - self.alpha) * self.vy
        return self.vx, self.vy

def find_largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
        return None
    return cnt

def skin_mask_from_bgr(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask

def main():
    cap = cv2.VideoCapture(CAPTURE_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return

    pos_history = deque(maxlen=POSITION_HISTORY)
    vel_filter = SmoothedVelocity()

    last_jump_time = 0.0
    last_slide_time = 0.0
    last_left_right_time = 0.0
    last_up_swipe_time = 0.0

    print("Starting. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_flipped = cv2.flip(frame, 1)
        h, w = frame_flipped.shape[:2]
        center_x = w // 2
        center_y = h // 2

        mask = skin_mask_from_bgr(frame_flipped)
        cnt = find_largest_contour(mask)

        centroid = None
        if cnt is not None:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
                pos_history.append((cx, cy, time.time()))
                cv2.drawContours(frame_flipped, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(frame_flipped, centroid, 6, (0, 0, 255), -1)

        vel_x = 0.0
        vel_y = 0.0
        if len(pos_history) >= 2:
            x0, y0, t0 = pos_history[0]
            x1, y1, t1 = pos_history[-1]
            dt = t1 - t0 if (t1 - t0) > 1e-6 else 1e-6
            raw_vx = (x1 - x0) / dt
            raw_vy = (y1 - y0) / dt
            vel_x, vel_y = vel_filter.update(raw_vx, raw_vy)
        else:
            vel_filter.initialized = False

        h_deadzone = int(H_DEADZONE_RATIO * w)
        left_zone = center_x - h_deadzone
        right_zone = center_x + h_deadzone

        now = time.time()
        if centroid is not None and now - last_left_right_time > H_MOVE_COOLDOWN:
            cx, cy = centroid
            if cx < left_zone:
                pyautogui.press(KEY_LEFT)
                last_left_right_time = now
                lr_text = "LEFT"
            elif cx > right_zone:
                pyautogui.press(KEY_RIGHT)
                last_left_right_time = now
                lr_text = "RIGHT"
            else:
                lr_text = "NEUTRAL"
        else:
            lr_text = "NO_HAND" if centroid is None else "WAIT"

        action_text = ""
        if centroid is not None:
            if vel_y < JUMP_VEL_THRESHOLD and (now - last_jump_time) > JUMP_COOLDOWN:
                pyautogui.press(KEY_JUMP)
                last_jump_time = now
                action_text = "JUMP"
                if (now - last_up_swipe_time) <= SLIDE_DOUBLE_TIME and (now - last_slide_time) > SLIDE_COOLDOWN:
                    pyautogui.press(KEY_SLIDE)
                    last_slide_time = now
                    action_text = "SLIDE (double-up)"
                last_up_swipe_time = now
            else:
                if vel_y < JUMP_VEL_THRESHOLD:
                    last_up_swipe_time = now

        if SHOW_DEBUG:
            cv2.line(frame_flipped, (center_x, 0), (center_x, h), (200, 200, 200), 1)
            cv2.line(frame_flipped, (left_zone, 0), (left_zone, h), (180, 180, 180), 1)
            cv2.line(frame_flipped, (right_zone, 0), (right_zone, h), (180, 180, 180), 1)
            vel_display = f"vel_x: {vel_x:.1f}px/s  vel_y: {vel_y:.1f}px/s"
            cv2.putText(frame_flipped, vel_display, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame_flipped, f"HORI: {lr_text}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
            cv2.putText(frame_flipped, f"ACTION: {action_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
            cv2.putText(frame_flipped, "Press 'q' to quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            small_mask = cv2.resize(mask_rgb, (160, 120))
            frame_flipped[0:120, w - 160:w] = small_mask

        cv2.imshow("One-Hand Subway Controller (DEBUG)", frame_flipped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
