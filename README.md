# -Object-Distance-Tracker-using-OpenCV
A real-time object distance tracking system built using Python and OpenCV. It detects objects based on their color and calculates the distance between them in centimeters (cm). The first detected object is automatically set as the **reference**, and all other objects are measured relative to it.
# 📏 Object Distance Tracker using OpenCV

A real-time object distance tracking system built using Python and OpenCV. It detects objects based on their color and calculates the distance between them in centimeters (cm). The first detected object is automatically set as the **reference**, and all other objects are measured relative to it.

---

## 🧠 Features

- 🎯 Detects multiple colored objects (red, green, blue, yellow, orange, purple).
- 📐 Measures distances **between objects** in **centimeters**.
- 📷 Uses webcam (or any connected camera) for live tracking.
- 🖱️ Auto-sets first detected object as the **reference object**.
- 🧾 Distance displayed with a line and label in real time.
- 🗂️ Optional calibration using an **A4 sheet** (21 cm width) for more accuracy.

---

## 🚀 How It Works

1. Launch the app.
2. The **first detected object** becomes the reference (shown with orange box).
3. Place other colored objects in view.
4. Distances between the reference and each object will be displayed live.
5. You can calibrate the scale using an A4 paper by pressing `c`.

---

## 🖥️ Demo

![Object Distance Tracker Demo](demo.gif)

> Note: Replace with your own screenshot or GIF if needed.

---

## 🧪 Controls

| Key | Action |
|-----|--------|
| `c` | Calibrate using A4 sheet (21 cm wide) |
| `r` | Reset reference and clear tracking |
| `q` | Quit the application |

---

## 🧰 Requirements

- Python 3.7+
- OpenCV
- NumPy

### Install dependencies:

```bash
pip install opencv-python numpy
