# AI Study Buddy & Focus Monitor

The AI Study Buddy is an intelligent desktop app that uses computer vision, acting as a real-time study companion. It uses an ordinary webcam to track focus levels, monitor blink rates, evaluate posture, detect note-taking behaviour, and maintain a structured log of study sessions.

---

# Summary

The **AI Study Buddy & Focus Monitor** addresses common digital study issues, such as drifting focus, slouched posture, and eye discomfort caused by hours in front of a screen.

Powered by **Google MediaPipe's 478-point Face Landmarker AI**, the system creates a live, detailed 3D surface map of the user's face to track their head orientation and eye positioning. Within just three seconds, it calibrates to the user's individual features, distinguishing between genuine distractions or bad posture and intentional, focused study behaviours (like looking down to take notes).

Reliability and user experience are at the core of the application’s design. The interface remains smooth and responsive at all times, while a structured session log- maintained seamlessly using `pandas` ensures users' progress and habits are always well documented.

---

## Key Features

### Real-Time Focus & Eye Care
* **Sci-Fi Scanner Overlay:** Renders a neon-green 478-point face landmark mesh directly over live webcam video offering instant visual feedback and precise tracking.
* **Blink Detection:** Monitors real-time blink rates, issuing alerts when blink frequency drops below healthy thresholds to help prevent dry eyes.
* **20-20-20 Rule Reminders:** Provides periodic notifications every 20 minutes to remind users to look away and rest their eyes in accordance with the 20-20-20 rule.
* **Distraction Alerts:** Delivers soft auditory alerts when eyes remain closed or the user turns away from the screen for prolonged periods, helping maintain focus and engagement.

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
