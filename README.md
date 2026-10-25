# AI Study Buddy & Focus Monitor

> **Zone 2: AI & Emerging Technologies** | *Tech Explore 2026-27*

An intelligent, real-time desktop application designed to improve student productivity and ergonomic health. Using a standard webcam and computer vision AI, the **AI Study Buddy** monitors focus, tracks eye-blink rates to prevent dry eyes, detects slouching/improper posture, and alerts the user to distractions.

---

## Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Key Features](#-key-features)
3. [Architecture & System Design](#-architecture--system-design)
4. [Quick Start & Setup](#-quick-start--setup)
5. [Presentation Track & Pitch Guide](#-presentation-track--pitch-guide)
6. [Viva / Technical Defense Notes](#-viva--technical-defense-notes)

---

## Executive Summary

### The Problem
During extended study sessions, students face three primary issues:
1. **Cognitive Distraction:** Head turns, looking away from study material, and screen-induced fatigue.
2. **Physical Strain & Ergonomics:** Slouching, leaning too close to the screen, or hunching forward, leading to neck and back pain.
3. **Computer Vision Syndrome (Eye Strain):** Decreased blink rate during screen focus causes dry eyes and strain.

### The Solution
The **AI Study Buddy & Focus Monitor** acts as an automated, privacy-first virtual study assistant. Operating entirely on local hardware, it captures real-time video feed via a webcam, maps **478 3D facial landmarks** using MediaPipe's latest Tasks API, and computes spatial metrics to evaluate posture, gaze direction, and eye aspect ratios (EAR).

---

## Key Features

- 🔬 **478-Point Sci-Fi Scanner Overlay:** Live facial mesh, iris tracking, and bounding brackets over a dark-themed UI built with CustomTkinter.
- 🎯 **Dynamic Focus Score (0–100%):** Decreases during distraction or drowsiness and recovers as deep work continues.
- 👁️ **Eye Care & Blink Rate Analytics:** Calculates Eye Aspect Ratio (EAR) and alerts the user if blinks-per-minute fall below healthy thresholds ($\approx 15\text{--}20\text{ blinks/min}$).
- 📏 **Ergonomic Posture Monitor:** Automatically calculates distance and baseline shifts to flag leaning in ($>118\%$ of standard size) or slumping ($>8\%$ drop in vertical baseline).
- ⏱️ **Active Study Timer:** Auto-pauses when the student leaves the desk and resumes upon detection.
- 🛡️ **Privacy First:** 100% local processing; video frames are processed on-the-fly and never recorded or uploaded.

---

## Architecture & System Design

```
+-----------------------------------------------------------------+
|                        GUI Thread (Tkinter)                     |
|  - Dashboard Updates (Focus, Blinks, Posture, Timers)            |
|  - Renders Sci-Fi Camera Feed via PIL/Image                     |
+-----------------------------------------------------------------+
                                ^
                                | (Thread-Safe Queue Mailbox)
                                v
+-----------------------------------------------------------------+
|                    Worker Thread (OpenCV & AI)                  |
|  - Captures Webcam Video Feed                                   |
|  - Executes MediaPipe FaceLandmarker Tasks API                  |
|  - Runs FocusBrain Logic (EAR, Baseline, Posture Shift)         |
|  - Renders Neon HUD Overlays on Frame                           |
+-----------------------------------------------------------------+
```

### Key Technical Concepts
1. **Separation of Concerns (OOP):**
   - `FocusBrain`: Pure mathematics and state management (decoupled from UI and hardware).
   - `StudyBuddyApp`: Event handling, multi-threading control, and UI rendering.
2. **Multithreaded Pipeline:** Heavy OpenCV frame acquisition and MediaPipe model inference run on a background daemon thread to prevent frame drops or UI freezing.
3. **Personal Calibration System:** Captures individual facial geometry during the initial $3.0\text{ s}$ buffer, adapting dynamically to various eye shapes, seating postures, and camera angles.

---

## Quick Start & Setup

### Prerequisites
- **Python:** 3.11 or 3.12 installed.
- **Hardware:** standard USB or integrated webcam.

### Installation & Execution

1. Clone or download the repository to your local machine.
2. Install required Python packages:
   ```bash
   pip install customtkinter opencv-python mediapipe pillow
   ```
3. Run the application:
   ```bash
   python study_buddy.py
   ```
   > *Note: On initial run, the application will automatically download the binary file (`face_landmarker.task`, ~3 MB) required by MediaPipe.*

---

## 🎤 Presentation Track & Pitch Guide

Use this timeline to structure your live project demonstration:

| Segment | Time | Content & Speaker Focus | Demonstration Action |
|---|---|---|---|
| **1. Hook & Problem** | 0:00–0:45 | Explain screen-fatigue, posture degradation, and distraction in digital learning. | Show initial app landing screen (Camera Off state). |
| **2. Solution & Launch** | 0:45–1:30 | Explain privacy-first approach and edge-AI processing. Click "Start Study Session". | Sit straight for $3\text{ s}$ during **Personal Calibration**. |
| **3. Live AI Capabilities** | 1:30–3:00 | Highlight the 478-point mesh, real-time EAR calculation, and live focus percentage. | **Blink fast/slowly** to change blink rate metrics. Turn head away to trigger **"LOOKING AWAY"**. |
| **4. Posture Engine** | 3:00–4:00 | Demonstrate spatial geometric monitoring. | **Lean into the screen** $\rightarrow$ Trigger "Leaning in too close".<br>**Slump in chair** $\rightarrow$ Trigger "Slumping down". |
| **5. Technical Architecture** | 4:00–5:00 | Detail OOP structure, background worker threads, and thread-safe queues. | Point out code structure (`FocusBrain` logic vs UI Thread). |

---

## Viva / Technical Defense Notes

### 1. Mathematical Formulas Used

#### Eye Aspect Ratio (EAR)
Computes eye openness by comparing vertical landmark distances to horizontal width:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

- **Open Eye:** $\text{EAR} \approx 0.25\text{--}0.35$
- **Closed Eye:** $\text{EAR} \le 0.15\text{--}0.21$

#### Head Pose & Turn Ratio
Determines horizontal nose offset relative to facial boundaries:
$$\text{Turn Ratio} = \frac{X_{\text{Nose}} - X_{\text{LeftCheek}}}{X_{\text{RightCheek}} - X_{\text{LeftCheek}}}$$
- Normal gaze points near $0.5$. Deviations beyond $\pm 0.17$ trigger a "Looking Away" state.

---

### 2. Anticipated Viva Questions & Model Answers

**Q1: Why did you use MediaPipe Tasks API instead of `mp.solutions`?**
> **Answer:** Google deprecated the legacy `mp.solutions` pipeline in newer releases (v0.10.3+). The `mediapipe.tasks.python.vision.FaceLandmarker` API provides higher execution speed, unified cross-platform C++ backend bindings, and optimized performance for 478-point 3D landmark localization.

**Q2: How do you prevent UI freezing while running heavy computer vision tasks?**
> **Answer:** We use Python’s `threading` module to separate the application into two tasks:
> - **Worker Thread:** Acquires webcam frames, runs MediaPipe inference, updates `FocusBrain` mathematical models, and draws HUD annotations.
> - **Main Thread:** Handles the CustomTkinter GUI loop. 
> Data transfers safely between threads via non-blocking instances of `queue.Queue`.

**Q3: How does the system adapt to different users with different facial features?**
> **Answer:** The app uses an automated calibration phase during the first $3.0\text{ seconds}$ of a session. It records baseline values for EAR, nose position, and facial width. The EAR blink threshold is set dynamically to $75\%$ of the user's specific baseline EAR rather than using a static global value.

**Q4: Is user video data saved or transmitted to an external server?**
> **Answer:** No. Image frames are processed in memory frame-by-frame and immediately overwritten. No video data or biometric records leave the local system memory.

**Q5: What happens if two people appear in front of the camera?**
> **Answer:** `FaceLandmarkerOptions` is explicitly configured with `num_faces=1`. The model isolates and tracks the primary detected face to ensure consistent metrics for one student.