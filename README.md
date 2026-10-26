# AI Study Buddy & Focus Monitor

An intelligent desktop app that uses computer vision, acting as a real-time study companion. It uses an ordinary webcam to track focus levels, monitor blink rates, evaluate posture, detect note-taking behavior, and maintain a structured log of study sessions.

---

# Summary

The **AI Study Buddy & Focus Monitor** addresses common digital study issues—such as loss of focus, poor posture, and dry eyes caused by reduced blink rates. 

By leveraging **Google MediaPipe's 478-point Face Landmarker AI**, the system creates a live 3D surface map of the user's face to track head orientations and eye positioning. It automatically calibrates to the individual user within the first 3 seconds of a session and intelligently distinguishes between harmful distraction/poor posture and intentional study habits (e.g., tipping the head down to write notes).

Designed with high reliability and stability in mind, the application operates on a robust structure to keep the UI smooth and responsive while maintaining a clean, structured session log via `pandas`.

---

## Key Features

### Real-Time Focus & Eye Care
* **Sci-Fi Scanner Overlay:** Renders a neon-green 478-point face landmark mesh directly over live webcam video.
* **Blink Detection:** Tracks real-time blinks and alerts users when their blink rate drops too low to prevent dry eyes.
* **20-20-20 Rule Reminders:** Issues periodic notifications every 20 minutes to remind users to look away and rest their eyes.
* **Distraction Alerts:** Triggers soft auditory alerts when eyes remain closed or the user turns away from the screen for prolonged periods.

### Adaptive Posture & Note-Taking Mode
* **Personalized Calibration:** Automatically learns the user's default eye shape, head angle, and posture in the first 3 seconds.
* **Posture Guard:** Detects excessive slouching or leaning in too close to the screen, reminding the user to sit upright.
* **Smart Note-Taking Detection:** Recognizes head pitch changes when a student tips their head down to read or write. It pauses posture alarms and adjusts eye-closure thresholds so note-taking isn't misidentified as distraction or drowsiness.

### Session Analytics
* **Live Focus Score:** Displays an interactive dynamic focus percentage based on engagement.
* **Single-Session Log Engine:** Log study time, focus level, total blinks, average blink rate, distractions, and posture alerts directly to `study_log.csv`.
* **Automatic Duplicate Cleanup:** Searches and eliminates duplicate records after every log export, archiving discarded rows safely in `study_log_duplicates.csv`.

---

## Frameworks Used

* **GUI Framework:** `customtkinter` (Modern Dark-Mode UI), `Pillow`
* **Computer Vision & AI:** `OpenCV`, `MediaPipe` (Tasks Vision API - 478-point Mesh)
* **Data Processing:** `pandas`, `numpy`
* **Concurrency:** `threading`, `queue.Queue` (Thread-safe communication between GUI & background worker)

---

## Getting Started

### Prerequisites
* **Python Version:** 3.11 or 3.12 recommended.
* **Hardware:** Any standard USB webcam or built-in laptop camera.

### Installation

1. **Clone or download** this repository.
2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Execute the Python script:
```bash
python study_buddy.py
```

> **Note:** On the first launch, the app will automatically download the lightweight MediaPipe model (`face_landmarker.task`, ~3 MB) into a local `model/` folder. All subsequent runs function completely **offline**.

---

## Key Concepts (Presentation & Viva Guide)

If presenting this project, consider highlighting these technical aspects:

1. **Object-Oriented Architecture:**
   * `FocusBrain`: Contains pure mathematical and analytical logic (EAR, pose tracking, pitch/yaw calculations). Independent of UI or hardware.
   * `SessionLog`: Handles data management, CSV I/O, and deduplication via `pandas`.
   * `StudyBuddyApp`: Manages GUI components, main loop, and thread safe queue rendering.
2. **Multi-Threading & Race-Condition Safety:** The camera and MediaPipe inference run on a dedicated background worker thread, feeding frames into a `queue.Queue` to keep the main GUI fluid at high frame rates.
3. **Adaptive Thresholding:** Rather than hardcoded values, posture and eye metrics are scaled against the initial 3-second baseline calibration.