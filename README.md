# 🕹️ Motion-Controlled Subway Surfers (One-Hand Tracking using OpenCV)

A fun and interactive computer vision project that lets you play **Subway Surfers** using just one hand and your webcam — no keyboard required!  
This project converts real-world motion into virtual game actions using **OpenCV**, **NumPy**, and **PyAutoGUI**.

---

## 🎮 Overview

Use your webcam to track hand movement in real time and convert gestures into game controls:

- ✋ Move Left / Right — by shifting your hand horizontally  
- ⬆️ Jump — with a single upward swipe  
- ⬆️⬆️ Slide — with a quick double upward swipe  

Each movement triggers a smooth, natural action for responsive gameplay.

---

## ⚙️ Tech Stack

| Tool / Library | Purpose |
|-----------------|----------|
| 🐍 Python | Core logic and gesture control |
| 🎥 OpenCV | Hand detection and tracking |
| ⌨️ PyAutoGUI | Sends keyboard inputs to the game |
| 🔢 NumPy | Fast array and velocity calculations |

---

## 🚀 Features

- ✅ Real-time hand tracking using only OpenCV (no Mediapipe)
- ✅ Velocity-based gesture detection for accurate jumps
- ✅ Neutral zones and cooldowns to avoid false triggers
- ✅ On-screen visual debugging (mask + motion info)
- ✅ Fully playable Subway Surfers with one-hand gestures

---

## 🧩 How It Works

1. Captures webcam feed and isolates the hand using **color segmentation (HSV)**  
2. Tracks the **centroid of the largest contour** (hand)  
3. Calculates hand **velocity and position** across frames  
4. Maps gestures:
   - Fast upward movement → Jump  
   - Two quick upward swipes → Slide  
   - Hand left/right of screen → Move Left / Right  
5. Sends corresponding **keyboard events via PyAutoGUI**

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Nileshpar835/Motion-Controlled-Subway-Surfers-with-One-Hand-Tracking.git
cd motion-controlled-subway-surfers
```

# Install dependencies
```
pip install opencv-python numpy pyautogui
```
## ▶️ Usage
 # Run the script:
```
python motion.py
```

- Position your webcam so your hand is clearly visible

- Open Subway Surfers (or any similar endless-runner game)

- Control the game using hand gestures! ✋

- Press ‘q’ to quit.

## 🧠 Future Enhancements

 - 🔮 Predictive smoothing for faster gesture response

 - 🖐️ Integration with Mediapipe for more robust hand tracking

 - 🎮 Support for more gesture-based games
