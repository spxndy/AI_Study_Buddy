"""
AI STUDY BUDDY & FOCUS MONITOR
==============================
Tech Explore 2026-27  |  Zone 2: AI & Emerging Technologies

What it does
------------
* Uses an ordinary webcam and Google's MediaPipe Face Landmarker AI, which finds
  478 points on your face in real time.
* Draws a neon-green "sci-fi scanner" face mesh over your live video.
* Counts your blinks and reminds you to blink if your eyes might be drying out.
* Shows a live Focus Level (%) that drops when you look away, walk away or doze off.
* Warns you when you lean in too close to the screen or slump down in your chair,
  but understands when you are just tipping your head down to write notes.
* Plays an alert sound when you get distracted.
* "Upload to Log" button: once a session has ENDED, adds ONE timestamped row (focus
  score, distractions, blinks, posture alerts, study time...) to a pandas DataFrame
  saved as study_log.csv. Only one log per session, none while a session is running,
  and the table is searched for duplicates after every log.

How to run
----------
1. Install Python 3.11 or 3.12.
2. In a terminal:   pip install customtkinter opencv-python mediapipe pillow pandas
3. Run:             python study_buddy.py
   The first run downloads the AI model file (a few MB) and saves it next to this
   script as "face_landmarker.task". After that it works offline.

Big ideas used (good to explain in your presentation / viva)
-------------------------------------------------------------
* Object-Oriented Programming: three classes.
      FocusBrain      -> the AI logic (pure maths, no camera, no window)
      SessionLog      -> the logbook: a pandas DataFrame saved to a CSV file
      StudyBuddyApp   -> the window, the buttons and the threads
* Multi-threading: the camera + AI run on a background "worker" thread, so the
  window never freezes.
* A Queue is the safe "mailbox" the worker uses to send pictures and stats to
  the window thread. (Tkinter windows must only be touched from the main thread.)
* Eye Aspect Ratio (EAR): a simple formula that tells open eyes from closed eyes.
* Personal calibration: the app learns YOUR normal posture and eye shape in the
  first 3 seconds, so it works for everyone.
* Note-taking detection: when your head tips down (nose moves lower on the face)
  the app switches to "writing mode": no posture alarms, and the eyes-closed test
  becomes stricter, because looking down at a page makes your eyelids droop.

Note: MediaPipe 0.10.31 and newer removed the old "mp.solutions" API that most
online tutorials use. This project uses the newer "Tasks" API, which works with
the current versions.
"""

import math
import os
import queue
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import customtkinter as ctk
from PIL import Image
from tkinter import messagebox

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

try:
    import winsound  # Windows only: plays a real beep through the speakers
except ImportError:
    winsound = None


# =============================================================================
# SETTINGS - try changing these during testing!
# =============================================================================
CAMERA_INDEX = 0  # 0 = built-in webcam. Try 1 for an external USB camera.
VIDEO_W, VIDEO_H = 560, 420  # size of the camera panel in the window

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model", "face_landmarker.task"
)
LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "log", "study_log.csv"
)  # the logbook is saved here
DUP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "log", "study_log_duplicates.csv"
)  # removed duplicates are kept here

CALIBRATION_SEC = 3.0  # learn your normal posture for this long at the start
NO_FACE_LIMIT_SEC = 1.5  # no face for this long  -> "No student detected" (was 3.0)
LOOK_AWAY_LIMIT_SEC = 1.5  # head turned sideways this long -> "Distracted!" (was 3.0)
DROWSY_LIMIT_SEC = 1.5  # eyes shut this long -> "Feeling sleepy?"
MAX_BLINK_SEC = 0.5  # an eye closure longer than this is not a blink
HEAD_TURN_LIMIT = 0.12  # how far the nose may drift sideways from normal (was 0.17)
HEAD_TURN_SMOOTH = 0.35  # smoothing of the head-turn number (1 = none, lower = calmer)
HEAD_TURN_RELEASE = (
    0.75  # you must come back within 75% of the limit to cancel the timer
)
FACE_LOST_GRACE_SEC = 0.4  # a face dropout shorter than this is ignored (camera blip)
FACE_REACQUIRE_SEC = (
    0.4  # a lost face must be seen steadily this long to count as 'back'
)
CUT_OFF_FRACTION = 0.08  # if more than 8% of the face points are outside the picture,
# the face is half out of view and is treated as 'not here'
LEAN_IN_RATIO = 1.25  # face 25% bigger than normal = leaning in too close
SLUMP_DROP = 0.08  # face dropped 8% of the frame height = slumping down
LEAN_HOLD_SEC = 8.0  # leaning in (head upright) must last this long before we warn
SLUMP_HOLD_SEC = 4.0  # slumping must last this long before we warn
WRITING_PITCH_DELTA = 0.07  # head tipped down by this much (see 'pitch') = taking notes
WRITING_EAR_FACTOR = 0.70  # while writing, eyes only count as 'closed' below
# this fraction of the normal EAR threshold
FOCUS_DROP_PER_SEC = 5.0  # focus % lost per second while distracted
FOCUS_RECOVER_PER_SEC = 1.0  # focus % regained per second while focused
LOW_BLINK_RATE = 10  # blinks per minute (a relaxed person blinks ~15-20)
BREAK_REMINDER_MIN = 20  # remind about the 20-20-20 eye rule every N minutes
ALERT_REPEAT_SEC = 8.0  # repeat the distraction beep this often

# Colours used by the dashboard
GREEN, AMBER, RED, GREY = "#1e8e4f", "#b9770e", "#b03a2e", "#3b3f45"
UPLOAD_FG = ("gray70", "gray25")  # normal colour of the Upload to Log button
BLUE = "#2e6da4"  # "taking notes" colour
NEON = "#00ff8c"

# =============================================================================
# FACE MAP - which of the 478 MediaPipe points we use
# =============================================================================
# Six points around each eye, in the order p1..p6 used by the EAR formula:
#   p1 = outer corner, p2/p3 = upper lid, p4 = inner corner, p5/p6 = lower lid
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
NOSE_BRIDGE, CHIN = 168, 152  # between the eyes, and bottom of the chin (for head tilt)
CHEEK_A, CHEEK_B = 234, 454  # left and right edges of the face


def _pairs(connections):
    """Turn MediaPipe's list of connections into a NumPy array of (start, end)."""
    return np.array([(c.start, c.end) for c in connections], dtype=np.int32)


_CONN = vision.FaceLandmarksConnections
MESH_LINES = _pairs(_CONN.FACE_LANDMARKS_TESSELATION)
EYE_LINES = _pairs(_CONN.FACE_LANDMARKS_LEFT_EYE + _CONN.FACE_LANDMARKS_RIGHT_EYE)
IRIS_LINES = _pairs(_CONN.FACE_LANDMARKS_LEFT_IRIS + _CONN.FACE_LANDMARKS_RIGHT_IRIS)
OVAL_LINES = _pairs(_CONN.FACE_LANDMARKS_FACE_OVAL)


# =============================================================================
# SMALL MATHS HELPERS
# =============================================================================
def distance(p, q):
    """Straight-line distance between two (x, y) points (Pythagoras)."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def eye_aspect_ratio(eye):
    """
    Eye Aspect Ratio (EAR) = how 'tall' the eye is compared with how 'wide' it is.

        EAR = ( |p2 - p6| + |p3 - p5| ) / ( 2 * |p1 - p4| )

    Open eye   -> about 0.25 to 0.35
    Closed eye -> close to 0.1 or less
    When it dips for a split second and comes back up, that's a blink.
    """
    p1, p2, p3, p4, p5, p6 = eye
    width = distance(p1, p4)
    if width < 1e-6:
        return 0.0
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * width)


def format_time(seconds):
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# =============================================================================
# CLASS 1: THE BRAIN (pure logic - no camera, no window)
# =============================================================================
class FocusBrain:
    """
    Feed it the face points once per video frame and it keeps track of
    focus, blinks and posture. Because it does not touch the camera or the
    window, it is easy to test on its own with made-up data.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # ---- the stats shown on the dashboard ("the memory") ----
        self.focus_score = 100.0  # starts at 100%, drops when distracted
        self.blink_count = 0  # total blinks this session
        self.is_slouching = False  # True when posture has been bad for a while
        self.posture_issue = None  # "leaning" or "slumping"
        self.is_distracted = False
        self.status = (
            "calibrating"  # calibrating/searching/away/looking_away/drowsy/focused
        )
        self.study_seconds = 0.0  # timer pauses when you are away
        self.focused_seconds = 0.0
        self.distraction_count = 0
        self.posture_alerts = 0
        self.ear = 0.0
        self.eyes_closed = False
        self.face_found = False
        self.is_writing = False  # True while the head is tipped down to write notes

        # ---- calibration: your personal "normal" ----
        self.baseline = None
        self.ear_threshold = 0.21
        self._calib_samples = []
        self._calib_start = None

        # ---- stopwatches for "how long has this been happening?" ----
        self._last_update = None
        self._last_face_time = None
        self._lost_for = (
            0.0  # 'leaky bucket': fills while no face, drains while a face is seen
        )
        self._visible_since = (
            None  # start of the current unbroken run of visible frames
        )
        self._present = False  # debounced 'the student is really here'
        self._turn_avg = None
        self._eye_closed_since = None
        self._turned_since = None
        self._posture_bad_since = None
        self._posture_issue_seen = None
        self._blink_times = deque()
        self._face_w_avg = None
        self._nose_y_avg = None
        self._pitch_avg = None

    # -------------------------------------------------------------------------
    def update(self, points, frame_w, frame_h, now):
        """
        points  : list of (x, y) face points from MediaPipe (0..1 scale), or None
        frame_w : picture width in pixels
        frame_h : picture height in pixels
        now     : current time in seconds (time.monotonic())
        """
        dt = 0.0 if self._last_update is None else min(now - self._last_update, 0.5)
        self._last_update = now
        if self._last_face_time is None:
            self._last_face_time = now

        # ---------- IS THE STUDENT REALLY HERE? (debounced) ----------
        # MediaPipe sometimes finds the face for a single frame and loses it again,
        # especially when it is half out of view. Without this check every blip
        # restarted the "no face" stopwatch, so "away" could never trigger.
        visible = bool(points) and len(points) >= 468 and not self._cut_off(points)
        # A "leaky bucket" does the counting: time without a face fills it, time
        # with a face drains it. One short blip barely matters, but a face that is
        # only seen now and then (half out of view) fills the bucket -> "gone".
        if visible:
            if self._visible_since is None:
                self._visible_since = now
            self._lost_for = max(0.0, self._lost_for - dt)
            if not self._present and now - self._visible_since >= FACE_REACQUIRE_SEC:
                self._present = True  # steadily visible -> really back
                self._lost_for = 0.0
        else:
            self._visible_since = None
            self._lost_for += dt
            if self._present and self._lost_for > FACE_LOST_GRACE_SEC:
                self._present = False  # gone for more than a blip

        # ---------- SAFEGUARD: nobody in front of the camera ----------
        if not self._present:
            self.face_found = False
            self.eyes_closed = False
            self._eye_closed_since = None
            self._turned_since = None
            self._posture_bad_since = None
            self._posture_issue_seen = None
            self.is_slouching = False
            self.is_writing = False
            if now - self._last_face_time >= NO_FACE_LIMIT_SEC:
                self.status = "away"
                self._set_distracted(True, dt)
            else:
                self.status = "searching"  # a few seconds of grace first
            return  # study timer is paused while nobody is here

        if not visible:
            return  # a tiny dropout (< FACE_LOST_GRACE_SEC): keep everything as it was

        self.face_found = True
        self._last_face_time = now

        def px(i):  # convert point i from 0..1 scale to pixels
            return (points[i][0] * frame_w, points[i][1] * frame_h)

        ear = (
            eye_aspect_ratio([px(i) for i in LEFT_EYE])
            + eye_aspect_ratio([px(i) for i in RIGHT_EYE])
        ) / 2.0
        self.ear = ear

        # Head turn: where is the nose between the two edges of the face?
        # 0.5 = facing the camera. Much smaller or bigger = head turned sideways.
        a, b, nose = px(CHEEK_A), px(CHEEK_B), px(NOSE_TIP)
        span = b[0] - a[0]
        head_turn = 0.5 if abs(span) < 1e-6 else (nose[0] - a[0]) / span

        # Head tilt ("pitch"): where does the nose tip sit between the bridge of the
        # nose (between the eyes) and the chin?  Tipping your head DOWN to write
        # pushes the nose tip lower on the face, so this number goes UP.  It does not
        # care how far away you are, so it separates "writing" from "slumping".
        bridge, chin = px(NOSE_BRIDGE), px(CHIN)
        face_len = chin[1] - bridge[1]
        pitch = 0.5 if abs(face_len) < 1e-6 else (nose[1] - bridge[1]) / face_len

        face_w = distance(a, b) / frame_w  # bigger face = closer to the screen
        nose_y = points[NOSE_TIP][1]  # bigger number = lower in the picture

        # ---------- CALIBRATION (first few seconds) ----------
        if self.baseline is None:
            if self._calib_start is None:
                self._calib_start = now
            self._calib_samples.append((ear, head_turn, face_w, nose_y, pitch))
            self.status = "calibrating"
            if (
                now - self._calib_start >= CALIBRATION_SEC
                and len(self._calib_samples) >= 10
            ):
                self._finish_calibration()
            return

        self.study_seconds += dt

        # ---------- 0. SMOOTHING + NOTE-TAKING CHECK ----------
        # Smooth the numbers so tiny wobbles don't cause false alarms
        self._face_w_avg = 0.85 * self._face_w_avg + 0.15 * face_w
        self._nose_y_avg = 0.85 * self._nose_y_avg + 0.15 * nose_y
        self._pitch_avg = 0.85 * self._pitch_avg + 0.15 * pitch
        self._turn_avg += HEAD_TURN_SMOOTH * (head_turn - self._turn_avg)
        writing = self._pitch_avg - self.baseline["pitch"] > WRITING_PITCH_DELTA
        if writing != self.is_writing:
            self._eye_closed_since = (
                None  # pose just changed: restart the eye stopwatch
            )
        self.is_writing = writing

        # ---------- 1. BLINKS ----------
        # Looking down at a page makes the eyelids droop, which lowers EAR even
        # though the eyes are open. So while writing, the eyes only count as
        # "closed" when they are REALLY shut (a stricter, lower threshold).
        threshold = self.ear_threshold * (WRITING_EAR_FACTOR if writing else 1.0)
        closed = ear < threshold
        closed_for = 0.0
        if closed:
            if self._eye_closed_since is None:
                self._eye_closed_since = now
            closed_for = now - self._eye_closed_since
        elif self._eye_closed_since is not None:
            # eyes just opened again: was it a quick blink or a long close?
            if now - self._eye_closed_since <= MAX_BLINK_SEC:
                self.blink_count += 1
                self._blink_times.append(now)
            self._eye_closed_since = None
        self.eyes_closed = closed
        drowsy = closed_for >= DROWSY_LIMIT_SEC

        # ---------- 2. LOOKING AWAY ----------
        # (Looking DOWN is fine - you might be reading a book or writing notes.)
        turn_off = abs(self._turn_avg - self.baseline["turn"])
        if turn_off > HEAD_TURN_LIMIT:
            if self._turned_since is None:
                self._turned_since = now
        elif turn_off < HEAD_TURN_LIMIT * HEAD_TURN_RELEASE:
            self._turned_since = None  # clearly facing the screen again
        # (in between: keep the stopwatch, so wobbling at the edge can't reset it)
        looking_away = (
            self._turned_since is not None
            and now - self._turned_since >= LOOK_AWAY_LIMIT_SEC
        )

        # ---------- 3. POSTURE ----------
        # Leaning over to write notes is NORMAL, so no posture alarms in writing mode.
        # Otherwise leaning in and slumping each get their own patience:
        # leaning (reading something small) is forgiven longer than slumping.
        leaning_in = self._face_w_avg > self.baseline["face_w"] * LEAN_IN_RATIO
        slumping = self._nose_y_avg - self.baseline["nose_y"] > SLUMP_DROP
        issue = (
            None
            if writing
            else ("leaning" if leaning_in else "slumping" if slumping else None)
        )
        if issue is None:
            self._posture_bad_since = None
            self._posture_issue_seen = None
            bad_for_a_while = False
        else:
            if issue != self._posture_issue_seen or self._posture_bad_since is None:
                self._posture_issue_seen = issue  # new problem: restart the stopwatch
                self._posture_bad_since = now
            hold = LEAN_HOLD_SEC if issue == "leaning" else SLUMP_HOLD_SEC
            bad_for_a_while = now - self._posture_bad_since >= hold
        if bad_for_a_while and not self.is_slouching:
            self.posture_alerts += 1
        self.is_slouching = bad_for_a_while
        self.posture_issue = issue

        # ---------- 4. FOCUS ----------
        if drowsy:
            self.status = "drowsy"
        elif looking_away:
            self.status = "looking_away"
        else:
            self.status = "focused"
            self.focused_seconds += dt
        self._set_distracted(drowsy or looking_away, dt)

    # -------------------------------------------------------------------------
    @staticmethod
    def _cut_off(points):
        """True when a good part of the face is outside the picture (half out of view)."""
        pts = np.asarray(points, dtype=np.float32)[:, :2]
        outside = (pts[:, 0] < 0) | (pts[:, 0] > 1) | (pts[:, 1] < 0) | (pts[:, 1] > 1)
        return float(outside.mean()) > CUT_OFF_FRACTION

    def _finish_calibration(self):
        ears, turns, widths, nose_ys, pitches = zip(*self._calib_samples)
        self.baseline = {
            "ear": float(np.median(ears)),  # median ignores the odd blink
            "turn": float(np.median(turns)),
            "face_w": float(np.median(widths)),
            "nose_y": float(np.median(nose_ys)),
            "pitch": float(np.median(pitches)),
        }
        # Personal blink threshold: 75% of YOUR normal open-eye EAR
        self.ear_threshold = min(max(0.75 * self.baseline["ear"], 0.15), 0.27)
        self._face_w_avg = self.baseline["face_w"]
        self._nose_y_avg = self.baseline["nose_y"]
        self._pitch_avg = self.baseline["pitch"]
        self._turn_avg = self.baseline["turn"]

    def _set_distracted(self, distracted, dt):
        if distracted and not self.is_distracted:
            self.distraction_count += 1
        self.is_distracted = distracted
        if distracted:
            self.focus_score -= FOCUS_DROP_PER_SEC * dt
        elif self.status == "focused":
            self.focus_score += FOCUS_RECOVER_PER_SEC * dt
        self.focus_score = max(0.0, min(100.0, self.focus_score))

    def blinks_per_minute(self, now):
        while self._blink_times and now - self._blink_times[0] > 60:
            self._blink_times.popleft()
        window = min(60.0, self.study_seconds)
        if window < 20:  # not enough data yet
            return None
        return len(self._blink_times) * 60.0 / window

    def snapshot(self, now):
        """A copy of the stats, safe to post to the window thread."""
        if self.baseline is None and self._calib_start is not None:
            calib_left = max(0.0, CALIBRATION_SEC - (now - self._calib_start))
        else:
            calib_left = CALIBRATION_SEC
        return {
            "status": self.status,
            "focus": self.focus_score,
            "blinks": self.blink_count,
            "blink_rate": self.blinks_per_minute(now),
            "slouching": self.is_slouching,
            "writing": self.is_writing,
            "posture_issue": self.posture_issue,
            "distracted": self.is_distracted,
            "study_seconds": self.study_seconds,
            "focused_seconds": self.focused_seconds,
            "distractions": self.distraction_count,
            "posture_alerts": self.posture_alerts,
            "eyes_closed": self.eyes_closed,
            "face_found": self.face_found,
            "calibrated": self.baseline is not None,
            "calib_left": calib_left,
            "fps": 0.0,
        }


# =============================================================================
# DRAWING THE SCI-FI SCANNER (runs on the worker thread, uses OpenCV)
# Colours are (Red, Green, Blue) because the picture is in RGB order.
# =============================================================================
HUD_COLOURS = {
    "focused": (0, 255, 140),
    "calibrating": (255, 200, 0),
    "searching": (255, 200, 0),
    "looking_away": (255, 70, 70),
    "drowsy": (255, 70, 70),
    "away": (255, 70, 70),
}
HUD_TEXT = {
    "focused": "FOCUS LOCKED",
    "calibrating": "CALIBRATING...",
    "searching": "SEARCHING...",
    "looking_away": "LOOKING AWAY",
    "drowsy": "EYES CLOSED",
    "away": "NO STUDENT DETECTED",
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, x, y, colour, scale=0.55):
    """Text with a dark box behind it so it is always readable."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x = int(max(0, min(x, img.shape[1] - tw - 8)))
    y = int(max(th + 8, min(y, img.shape[0] - 2)))
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y + base - 2), (10, 14, 12), -1)
    cv2.putText(img, text, (x + 4, y - 4), FONT, scale, colour, 1, cv2.LINE_AA)


def _centred_text(img, text, y, colour, scale, thickness=2):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(
        img,
        text,
        ((img.shape[1] - tw) // 2, y),
        FONT,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _corner_brackets(img, x1, y1, x2, y2, colour, length=22, thickness=3):
    for cx, cy, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), colour, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), colour, thickness, cv2.LINE_AA)


def draw_scanner_hud(img, points, snap, now):
    """Draws the neon face mesh and status labels straight onto img (RGB)."""
    h, w = img.shape[:2]
    colour = HUD_COLOURS.get(snap["status"], (0, 255, 140))

    if points:
        P = (np.asarray(points, dtype=np.float32)[:, :2] * [w, h]).astype(np.int32)

        # 1) the connect-the-dots mesh, blended in so it glows softly
        overlay = img.copy()
        cv2.polylines(
            overlay, list(P[MESH_LINES]), False, (0, 255, 140), 1, cv2.LINE_AA
        )
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)

        # 2) eyes (cyan when open, red flash when closed) and irises
        eye_colour = (255, 70, 70) if snap["eyes_closed"] else (0, 230, 255)
        cv2.polylines(img, list(P[EYE_LINES]), False, eye_colour, 1, cv2.LINE_AA)
        if len(P) > IRIS_LINES.max():
            cv2.polylines(
                img, list(P[IRIS_LINES]), False, (255, 255, 255), 1, cv2.LINE_AA
            )

        # 3) face outline in the status colour
        cv2.polylines(img, list(P[OVAL_LINES]), False, colour, 2, cv2.LINE_AA)

        # 4) target brackets + a moving scan line
        x1, y1 = np.clip(P.min(axis=0) - 18, 0, [w - 1, h - 1])
        x2, y2 = np.clip(P.max(axis=0) + 18, 0, [w - 1, h - 1])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        _corner_brackets(img, x1, y1, x2, y2, colour)
        sweep = (math.sin(now * 2.5) + 1) / 2
        sy = int(y1 + sweep * (y2 - y1))
        cv2.line(img, (x1, sy), (x2, sy), colour, 1, cv2.LINE_AA)

        # status label above the face (or below it if the face fills the top)
        label_y = y1 - 6 if y1 > 60 else min(y2 + 26, h - 30)
        text = HUD_TEXT.get(snap["status"], "")
        if snap.get("writing") and snap["status"] == "focused":
            text = "NOTE-TAKING"
        _label(img, text, x1, label_y, colour)
    else:
        # Nobody there: dim the picture and show a warning
        img[:] = cv2.convertScaleAbs(img, alpha=0.45)
        if snap["status"] == "away":
            _centred_text(img, "NO STUDENT DETECTED", h // 2, (255, 70, 70), 0.9)
            _centred_text(img, "timer paused", h // 2 + 34, (220, 220, 220), 0.6, 1)
        else:
            _centred_text(img, "SEARCHING FOR FACE...", h // 2, (255, 200, 0), 0.8)

    _label(img, "AI STUDY BUDDY  |  478-POINT FACE MESH", 8, 26, (0, 255, 140), 0.45)
    if snap.get("fps"):
        _label(img, f"{snap['fps']:.0f} FPS", 8, h - 8, (180, 180, 180), 0.45)


def make_placeholder(lines):
    """A dark 'screen off' picture with a grid, used when the camera is off."""
    img = np.full((VIDEO_H, VIDEO_W, 3), (16, 20, 24), dtype=np.uint8)
    for x in range(0, VIDEO_W, 40):
        cv2.line(img, (x, 0), (x, VIDEO_H), (22, 38, 32), 1)
    for y in range(0, VIDEO_H, 40):
        cv2.line(img, (0, y), (VIDEO_W, y), (22, 38, 32), 1)
    y = VIDEO_H // 2 - (len(lines) - 1) * 22
    for text, scale, colour in lines:
        _centred_text(img, text, y, colour, scale, 2 if scale >= 0.8 else 1)
        y += 44
    return Image.fromarray(img)


# =============================================================================
# CAMERA + MODEL HELPERS
# =============================================================================
def download_model():
    tmp = MODEL_PATH + ".download"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_PATH)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            "Couldn't download the AI model. Connect to the internet and try "
            "again, or download it in a browser from:\n"
            + MODEL_URL
            + "\nand save it next to study_buddy.py as face_landmarker.task"
        ) from exc


def open_camera(index):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# =============================================================================
# CLASS 2: THE LOGBOOK (a pandas DataFrame that is also saved to a CSV file)
# =============================================================================
class SessionLog:
    """
    The logbook. When a session has ended, "Upload to Log" adds ONE timestamped row
    (a summary of that session) to a pandas DataFrame (self.df), which is saved to
    study_log.csv so the history survives restarts.

    Rules enforced here:
      * ONE row per session (a session is identified by its start time).
      * After every log the whole table is searched for duplicates and cleaned:
          - identical rows                  -> keep the first copy
          - several rows for one session    -> keep the latest (most complete) one
        Removed rows are not thrown away; they are archived in study_log_duplicates.csv.
    """

    COLUMNS = [
        "timestamp",
        "session_start",
        "study_seconds",
        "focused_seconds",
        "focus_score_pct",
        "focused_time_pct",
        "blinks",
        "blinks_per_min",
        "distractions",
        "posture_alerts",
    ]

    def __init__(self, path=LOG_PATH, dup_path=DUP_PATH):
        self.path = path
        self.dup_path = dup_path
        self.df = self._load()
        self.cleaned = 0  # duplicate rows removed since the last successful save
        self._pending_archive = []  # removed rows waiting to be written to dup_path

    def _load(self):
        if os.path.exists(self.path):
            try:
                df = pd.read_csv(self.path, parse_dates=["timestamp", "session_start"])
                return df.reindex(columns=self.COLUMNS)
            except Exception:
                # Unreadable file: keep it as a backup instead of overwriting it
                os.replace(self.path, self.path + ".bad")
        return pd.DataFrame(columns=self.COLUMNS)

    def has_session(self, session_start):
        """Is there already a row for the session that started at this time?"""
        return bool((self.df["session_start"] == session_start).any())

    def add(self, snap, session_start):
        """
        Add one timestamped row built from a stats snapshot (in memory only), then
        search for duplicates. Returns True if a row was added, or False if this
        session is already in the log (the one-log-per-session rule).
        """
        added = not self.has_session(session_start)
        if added:
            study = snap["study_seconds"]
            rate = snap["blink_rate"]
            row = {
                "timestamp": datetime.now().replace(microsecond=0),
                "session_start": session_start,
                "study_seconds": round(study, 1),
                "focused_seconds": round(snap["focused_seconds"], 1),
                "focus_score_pct": round(snap["focus"], 1),
                "focused_time_pct": round(100.0 * snap["focused_seconds"] / study, 1)
                if study > 0
                else 0.0,
                "blinks": snap["blinks"],
                "blinks_per_min": float("nan") if rate is None else round(rate, 1),
                "distractions": snap["distractions"],
                "posture_alerts": snap["posture_alerts"],
            }
            new_row = pd.DataFrame([row], columns=self.COLUMNS)
            # (starting from an empty table avoids a pandas warning about empty concat)
            self.df = (
                new_row
                if self.df.empty
                else pd.concat([self.df, new_row], ignore_index=True)
            )
        self.remove_duplicates()  # automatic duplicate search after every log
        return added

    def remove_duplicates(self):
        """Search the table for duplicate rows and remove them. Returns the removed rows."""
        if len(self.df) < 2:
            return self.df.iloc[0:0]
        # Step 1: rows that are identical in every column (keep the first copy)
        identical = self.df.duplicated(keep="first")
        # Step 2: of the rest, several rows for the same session (keep the latest one)
        rest = self.df[~identical]
        older = (
            rest.duplicated(subset=["session_start"], keep="last")
            & rest["session_start"].notna()
        )
        drop = list(self.df.index[identical]) + list(rest.index[older])
        removed = self.df.loc[drop]
        if len(removed):
            self.df = self.df.drop(index=drop).reset_index(drop=True)
            self.cleaned += len(removed)
            self._pending_archive.append(removed)
        return removed

    def save(self):
        """Write the table to the CSV (and archive removed duplicates). Raises if the file is locked."""
        tmp = self.path + ".tmp"
        try:
            self.df.to_csv(tmp, index=False)
            os.replace(tmp, self.path)  # swap in only when fully written
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        if self._pending_archive:  # only after the main file is safely written
            archive = pd.concat(self._pending_archive, ignore_index=True)
            archive.to_csv(
                self.dup_path,
                mode="a",
                index=False,
                header=not os.path.exists(self.dup_path),
            )
            self._pending_archive = []
        self.cleaned = 0


# =============================================================================
# CLASS 3: THE APP (window, buttons and the two threads)
# =============================================================================
class StudyBuddyApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Study Buddy & Focus Monitor")
        self.geometry("1080x640")
        self.minsize(1000, 600)

        self.brain = FocusBrain()
        self.log = SessionLog()
        self._final_snap = None  # stats captured when the session stopped
        self._session_started_at = None
        self._session_logged = False  # True once THIS session has been logged
        self._upload_job = None

        # Thread-to-thread "mailboxes"
        self.frame_queue = queue.Queue(maxsize=1)  # only the newest picture matters
        self.msg_queue = queue.Queue()  # status messages, never dropped
        self.stop_event = threading.Event()  # the worker's "please stop" flag
        self.worker = None
        self.running = False
        self._session_id = 0
        self._last_error = None

        # For sounds and tips
        self._prev_distracted = False
        self._prev_slouching = False
        self._last_alert = 0.0
        self._break_reminders = 0
        self._break_tip_until = 0.0

        self._ui_cache = {}
        self._video_img = None
        self._after_id = None

        self._build_layout()
        self._show_placeholder(
            [
                ("AI STUDY BUDDY", 1.0, (0, 255, 140)),
                ("Press 'Start Study Session'", 0.6, (200, 200, 200)),
                ("to switch on the AI scanner", 0.6, (200, 200, 200)),
            ]
        )

        # Clicking the window's X button calls on_close (clean shutdown)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._after_id = self.after(30, self._refresh_ui)

    # -------------------------------------------------------------------------
    # LAYOUT
    # -------------------------------------------------------------------------
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)  # camera panel stretches
        self.grid_columnconfigure(1, weight=0)  # dashboard keeps a fixed width
        self.grid_rowconfigure(0, weight=1)

        # ---------- LEFT: the AI vision canvas ----------
        left = ctk.CTkFrame(self, corner_radius=16)
        left.grid(row=0, column=0, padx=(18, 9), pady=18, sticky="nsew")
        ctk.CTkLabel(
            left,
            text="AI VISION",
            text_color=NEON,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(12, 0))
        self.video_label = ctk.CTkLabel(left, text="")
        self.video_label.pack(expand=True, padx=12, pady=8)
        self.message_label = ctk.CTkLabel(
            left,
            text="Camera is off",
            text_color="gray65",
            wraplength=VIDEO_W,
            justify="center",
        )
        self.message_label.pack(pady=(0, 12), padx=12)

        # ---------- RIGHT: the stats dashboard ----------
        right = ctk.CTkFrame(self, corner_radius=16, width=420)
        right.grid(row=0, column=1, padx=(9, 18), pady=18, sticky="nsew")
        right.grid_propagate(False)  # long text can't stretch or shift the panel
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right, text="AI Study Buddy", font=ctk.CTkFont(size=26, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=(16, 0), sticky="w")
        ctk.CTkLabel(
            right,
            text="Focus, blinks and posture - live",
            text_color="gray60",
            font=ctk.CTkFont(size=13),
        ).grid(row=1, column=0, padx=20, sticky="w")

        # Focus level (big number + bar)
        focus_card = ctk.CTkFrame(
            right, corner_radius=12, fg_color=("gray85", "gray17")
        )
        focus_card.grid(row=2, column=0, padx=16, pady=(12, 6), sticky="ew")
        ctk.CTkLabel(
            focus_card,
            text="FOCUS LEVEL",
            text_color="gray60",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(pady=(8, 0))
        self.focus_value = ctk.CTkLabel(
            focus_card, text="100%", font=ctk.CTkFont(size=54, weight="bold")
        )
        self.focus_value.pack()
        self.focus_bar = ctk.CTkProgressBar(focus_card, height=12)
        self.focus_bar.pack(fill="x", padx=18, pady=(0, 12))
        self.focus_bar.set(1.0)

        # Three small stat cards
        stats = ctk.CTkFrame(right, fg_color="transparent")
        stats.grid(row=3, column=0, padx=10, pady=6, sticky="ew")
        for col in range(3):
            stats.grid_columnconfigure(col, weight=1, uniform="stats")
        self.blink_value, self.blink_sub = self._stat_card(stats, 0, "BLINKS")
        self.time_value, self.time_sub = self._stat_card(stats, 1, "STUDY TIME")
        self.distract_value, self.distract_sub = self._stat_card(
            stats, 2, "DISTRACTIONS"
        )
        self.blink_sub.configure(text="-- /min")
        self.time_sub.configure(text="not started")
        self.distract_sub.configure(text="this session")

        # Posture alert box (green / red)
        self.posture_box = ctk.CTkFrame(right, corner_radius=10, fg_color=GREY)
        self.posture_box.grid(row=4, column=0, padx=16, pady=(8, 4), sticky="ew")
        self.posture_label = ctk.CTkLabel(
            self.posture_box,
            text="Posture: waiting...",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.posture_label.pack(pady=9)

        # Focus status box
        self.status_box = ctk.CTkFrame(right, corner_radius=10, fg_color=GREY)
        self.status_box.grid(row=5, column=0, padx=16, pady=4, sticky="ew")
        self.status_label = ctk.CTkLabel(
            self.status_box,
            text="Ready when you are",
            font=ctk.CTkFont(size=15, weight="bold"),
            wraplength=380,
        )
        self.status_label.pack(pady=9, padx=8)

        # Eye-care tip / session summary
        self.tip_label = ctk.CTkLabel(
            right,
            text="Tip: keep the screen about an arm's length away.",
            text_color="gray70",
            wraplength=390,
            justify="left",
            font=ctk.CTkFont(size=13),
        )
        self.tip_label.grid(row=6, column=0, padx=20, pady=(6, 4), sticky="w")

        options = ctk.CTkFrame(right, fg_color="transparent")
        options.grid(row=7, column=0, padx=20, pady=4, sticky="ew")
        options.grid_columnconfigure(0, weight=1)
        self.sound_switch = ctk.CTkSwitch(options, text="Alert sounds")
        self.sound_switch.grid(row=0, column=0, sticky="w")
        self.sound_switch.select()
        self.upload_button = ctk.CTkButton(
            options,
            text="Upload to Log",
            width=150,
            height=30,
            state="disabled",
            corner_radius=8,
            fg_color=UPLOAD_FG,
            hover_color=("gray60", "gray30"),
            command=self.upload_to_log,
        )
        self.upload_button.grid(row=0, column=1, sticky="e")

        right.grid_rowconfigure(8, weight=1)  # pushes the button to the bottom
        self.start_button = ctk.CTkButton(
            right,
            text="Start Study Session",
            height=52,
            corner_radius=12,
            font=ctk.CTkFont(size=19, weight="bold"),
            command=self.toggle_session,
        )
        self.start_button.grid(row=9, column=0, padx=16, pady=(6, 16), sticky="ew")
        self._button_colours = (
            self.start_button.cget("fg_color"),
            self.start_button.cget("hover_color"),
        )

    def _stat_card(self, parent, col, title):
        card = ctk.CTkFrame(parent, corner_radius=12, fg_color=("gray85", "gray17"))
        card.grid(row=0, column=col, padx=6, sticky="nsew")
        ctk.CTkLabel(
            card,
            text=title,
            text_color="gray60",
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(pady=(8, 0))
        value = ctk.CTkLabel(card, text="0", font=ctk.CTkFont(size=26, weight="bold"))
        value.pack()
        sub = ctk.CTkLabel(
            card, text="", text_color="gray60", font=ctk.CTkFont(size=11)
        )
        sub.pack(pady=(0, 8))
        return value, sub

    def _set(self, widget, **options):
        """Only redraw a widget when something actually changed (keeps it smooth)."""
        key = id(widget)
        if self._ui_cache.get(key) != options:
            widget.configure(**options)
            self._ui_cache[key] = options

    def _show_placeholder(self, lines):
        img = make_placeholder(lines)
        self._video_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.video_label.configure(image=self._video_img)

    # -------------------------------------------------------------------------
    # START / STOP
    # -------------------------------------------------------------------------
    def toggle_session(self):
        if self.running:
            self.stop_session()
        else:
            self.start_session()

    def start_session(self):
        self.brain.reset()
        self.stop_event.clear()
        self._session_id += 1
        self._last_error = None
        self._final_snap = None
        self._session_logged = False
        self._session_started_at = datetime.now().replace(microsecond=0)
        self._prev_distracted = self._prev_slouching = False
        self._last_alert = 0.0
        self._break_reminders = 0
        self._break_tip_until = 0.0
        self.running = True
        self._refresh_upload_button()  # locked while the session runs

        self.start_button.configure(
            text="Stop Session", fg_color=RED, hover_color="#922b21"
        )
        self._set(self.status_box, fg_color=AMBER)
        self._set(self.status_label, text="Starting up...")
        self._set(
            self.tip_label,
            text="Sit up straight and look at the screen for 3 seconds "
            "while I learn your normal posture.",
        )

        # Thread 2: the background worker (camera + AI)
        self.worker = threading.Thread(
            target=self._camera_worker, args=(self._session_id,), daemon=True
        )
        self.worker.start()

    def stop_session(self):
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=3)
        self.worker = None
        self.running = False
        # Worker has stopped, so it is now safe to read the final numbers.
        self._final_snap = self.brain.snapshot(time.monotonic())
        self._refresh_upload_button()  # unlocked: this session can now be logged once
        fg, hover = self._button_colours
        self.start_button.configure(
            text="Start Study Session", fg_color=fg, hover_color=hover
        )

        b = self.brain
        self._set(self.posture_box, fg_color=GREY)
        self._set(self.posture_label, text="Posture: --")
        if self._last_error:
            self._set(self.status_box, fg_color=RED)
            self._set(
                self.status_label, text="Stopped - see the message under the camera"
            )
            self._set(self.message_label, text=self._last_error)
            self._set(self.tip_label, text="Fix the problem, then press Start again.")
            self._show_placeholder(
                [
                    ("CAMERA / AI ERROR", 0.9, (255, 70, 70)),
                    ("See the message below", 0.6, (200, 200, 200)),
                ]
            )
            return

        rate = b.blinks_per_minute(time.monotonic())
        rate_text = f" ({rate:.0f}/min)" if rate is not None else ""
        self._set(self.status_box, fg_color=GREY)
        self._set(self.status_label, text="Session complete - nice work!")
        self._set(
            self.tip_label,
            text=(
                f"Studied {format_time(b.study_seconds)} (focused {format_time(b.focused_seconds)})"
                f"  |  Final focus {b.focus_score:.0f}%\n"
                f"Blinks {b.blink_count}{rate_text}  |  Distractions {b.distraction_count}"
                f"  |  Posture alerts {b.posture_alerts}"
            ),
        )
        self._set(self.time_sub, text="finished")
        self._set(self.message_label, text="Camera is off")
        self._show_placeholder(
            [
                ("SESSION COMPLETE", 0.9, (0, 255, 140)),
                ("Your summary is on the right", 0.6, (200, 200, 200)),
            ]
        )

    # -------------------------------------------------------------------------
    # UPLOAD TO LOG (button): ONE timestamped row per FINISHED session
    # -------------------------------------------------------------------------
    def upload_to_log(self):
        # The button is already disabled in these cases; these checks are a second lock.
        if self.running:  # rule 1: never in the middle of a session
            return
        if self._session_logged:  # rule 2: only one log per session
            return
        snap = self._final_snap  # the numbers captured when the session stopped
        if snap is None or not snap["calibrated"] or snap["study_seconds"] <= 0:
            return  # nothing worth logging
        try:
            self.log.add(snap, self._session_started_at)  # also searches for duplicates
            cleaned = self.log.cleaned
            self.log.save()
        except Exception as exc:  # e.g. the CSV is open in Excel
            self._flash_upload_button("Save failed")
            messagebox.showerror(
                "Couldn't save the log",
                f"Could not write {os.path.basename(LOG_PATH)}:\n{exc}\n\n"
                "If the file is open in Excel, close it and press Upload to Log again. "
                "Your row is kept in memory and will not be added twice.",
            )
            return
        self._session_logged = True
        self._refresh_upload_button()  # now shows "Logged" (locked)
        plural = "" if cleaned == 1 else "s"
        self._set(
            self.status_label,
            text="Logged - no duplicates found"
            if cleaned == 0
            else f"Logged - removed {cleaned} duplicate row{plural}",
        )
        print(self.log.df.tail())  # peek at the DataFrame in the terminal
        if cleaned:
            print(f"Removed {cleaned} duplicate row(s); copies kept in {DUP_PATH}")

    def _refresh_upload_button(self):
        """Set the button's text, colour and enabled/disabled state from the situation."""
        if self._upload_job is not None:
            self.after_cancel(self._upload_job)
            self._upload_job = None
        snap = self._final_snap
        has_data = snap is not None and snap["calibrated"] and snap["study_seconds"] > 0
        if self.running:
            text, colour, enabled = "Log after session", UPLOAD_FG, False
        elif self._session_logged:
            text, colour, enabled = "Logged", GREEN, False
        else:
            text, colour, enabled = "Upload to Log", UPLOAD_FG, has_data
        self.upload_button.configure(
            text=text, fg_color=colour, state="normal" if enabled else "disabled"
        )

    def _flash_upload_button(self, text):
        """Show a short red message on the button, then restore it."""
        if self._upload_job is not None:
            self.after_cancel(self._upload_job)
        self.upload_button.configure(text=text, fg_color=RED)
        self._upload_job = self.after(2000, self._refresh_upload_button)

    # -------------------------------------------------------------------------
    # THREAD 2: the background worker. It must NEVER touch the window directly.
    # -------------------------------------------------------------------------
    def _camera_worker(self, session_id):
        cap = None
        landmarker = None
        try:
            if not os.path.exists(MODEL_PATH):
                self._post(
                    session_id,
                    "msg",
                    "Downloading the AI face model (first run only)...",
                )
                download_model()

            self._post(session_id, "msg", "Loading the AI...")
            options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)

            self._post(session_id, "msg", "Switching on the camera...")
            cap = open_camera(CAMERA_INDEX)
            if cap is None:
                raise RuntimeError(
                    "Could not open the webcam. Is another app "
                    "(Zoom, Teams, Camera) using it?"
                )
            self._post(session_id, "msg", "AI scanner running - camera light is ON")

            last_ts = -1
            bad_reads = 0
            reopen_tries = 0
            fps = 0.0
            last_frame_time = time.monotonic()

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    bad_reads += 1
                    if bad_reads > 60:
                        # The camera went quiet. Try re-opening it a few times
                        # before giving up (some webcams freeze for a moment).
                        reopen_tries += 1
                        if reopen_tries > 3:
                            raise RuntimeError("The webcam stopped sending pictures.")
                        cap.release()
                        time.sleep(0.5)
                        cap = open_camera(CAMERA_INDEX)
                        if cap is None:
                            raise RuntimeError("The webcam stopped sending pictures.")
                        bad_reads = 0
                    time.sleep(0.03)
                    continue
                bad_reads = 0
                reopen_tries = 0

                frame = cv2.flip(frame, 1)  # mirror image, feels natural
                h, w = frame.shape[:2]
                if w != VIDEO_W:
                    frame = cv2.resize(
                        frame,
                        (VIDEO_W, int(h * VIDEO_W / w)),
                        interpolation=cv2.INTER_AREA,
                    )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]

                # MediaPipe needs a timestamp that always goes up
                ts = int(time.monotonic() * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
                )

                points = None
                if result.face_landmarks:
                    points = [(lm.x, lm.y) for lm in result.face_landmarks[0]]

                now = time.monotonic()
                self.brain.update(points, w, h, now)
                snap = self.brain.snapshot(now)

                frame_time = now - last_frame_time
                last_frame_time = now
                if frame_time > 0:
                    fps = (
                        0.9 * fps + 0.1 * (1.0 / frame_time)
                        if fps
                        else 1.0 / frame_time
                    )
                snap["fps"] = fps

                draw_scanner_hud(rgb, points, snap, now)
                self._push_frame(session_id, rgb, snap)

        except Exception as exc:
            self._post(session_id, "error", str(exc))
        finally:
            # SAFEGUARD: always release the camera so the webcam light turns off
            if cap is not None:
                cap.release()
            if landmarker is not None:
                landmarker.close()
            self._post(session_id, "stopped", None)

    def _post(self, session_id, kind, payload):
        self.msg_queue.put((session_id, kind, payload))

    def _push_frame(self, session_id, frame, snap):
        try:
            self.frame_queue.get_nowait()  # throw away an old picture
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait((session_id, frame, snap))
        except queue.Full:
            pass

    # -------------------------------------------------------------------------
    # THREAD 1: the main thread checks the mailbox ~30 times a second
    # -------------------------------------------------------------------------
    def _refresh_ui(self):
        # 1) status messages from the worker
        while True:
            try:
                sid, kind, payload = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            if sid != self._session_id:
                continue  # leftover from an older session
            if kind == "msg":
                self._set(self.message_label, text=payload)
            elif kind == "error":
                self._last_error = payload
            elif kind == "stopped" and self.running:
                self.stop_session()  # worker ended on its own (e.g. an error)

        # 2) the newest picture + stats
        try:
            sid, frame, snap = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            if sid == self._session_id and self.running:
                img = Image.fromarray(frame)
                self._video_img = ctk.CTkImage(
                    light_image=img, dark_image=img, size=img.size
                )
                self.video_label.configure(image=self._video_img)
                self._update_dashboard(snap)

        self._after_id = self.after(30, self._refresh_ui)

    def _update_dashboard(self, snap):
        now = time.monotonic()

        # Focus level
        focus = snap["focus"]
        colour = "#2ecc71" if focus >= 70 else "#f5b041" if focus >= 40 else "#ff5c5c"
        self._set(self.focus_value, text=f"{focus:.0f}%", text_color=colour)
        self._set(self.focus_bar, progress_color=colour)
        self.focus_bar.set(focus / 100.0)

        # Stat cards
        rate = snap["blink_rate"]
        self._set(self.blink_value, text=str(snap["blinks"]))
        self._set(
            self.blink_sub, text="-- /min" if rate is None else f"{rate:.0f} /min"
        )
        self._set(self.time_value, text=format_time(snap["study_seconds"]))
        paused = snap["status"] in ("away", "searching")
        self._set(
            self.time_sub,
            text="paused"
            if paused
            else "calibrating"
            if not snap["calibrated"]
            else "studying",
        )
        self._set(self.distract_value, text=str(snap["distractions"]))

        # Posture box
        if not snap["calibrated"] or not snap["face_found"]:
            self._set(self.posture_box, fg_color=GREY)
            self._set(self.posture_label, text="Posture: waiting...")
        elif snap["writing"]:
            self._set(self.posture_box, fg_color=BLUE)
            self._set(self.posture_label, text="Taking notes - posture check paused")
        elif snap["slouching"]:
            why = (
                "You're leaning in too close"
                if snap["posture_issue"] == "leaning"
                else "You're slumping down"
            )
            self._set(self.posture_box, fg_color=RED)
            self._set(self.posture_label, text=f"Sit up straight! {why}")
        else:
            self._set(self.posture_box, fg_color=GREEN)
            self._set(self.posture_label, text="Great Posture!")

        # Focus status box
        status = snap["status"]
        messages = {
            "calibrating": (
                AMBER,
                f"Calibrating... sit up straight and look at the "
                f"screen ({math.ceil(snap['calib_left'])})",
            ),
            "searching": (AMBER, "Looking for you..."),
            "away": (RED, "No student detected... pausing timer"),
            "looking_away": (RED, "Distracted! Eyes back on your work"),
            "drowsy": (RED, "Eyes closed - feeling sleepy? Take a short break"),
            "focused": (GREEN, "Focused - keep going!"),
        }
        box_colour, text = messages.get(status, (GREY, ""))
        self._set(self.status_box, fg_color=box_colour)
        self._set(self.status_label, text=text)

        # Sounds
        if snap["distracted"] and (
            not self._prev_distracted or now - self._last_alert >= ALERT_REPEAT_SEC
        ):
            self._play_alert("distracted")
            self._last_alert = now
        if snap["slouching"] and not self._prev_slouching:
            self._play_alert("posture")
        self._prev_distracted = snap["distracted"]
        self._prev_slouching = snap["slouching"]

        # Eye-care tips
        due = int(snap["study_seconds"] // (BREAK_REMINDER_MIN * 60))
        if due > self._break_reminders:
            self._break_reminders = due
            self._break_tip_until = now + 30
            self._play_alert("break")
        if not snap["calibrated"]:
            tip = "Learning your normal posture and eye shape..."
        elif now < self._break_tip_until:
            tip = (
                "20-20-20 break: look at something 20 feet away for 20 seconds "
                "to rest your eyes."
            )
        elif rate is not None and rate < LOW_BLINK_RATE:
            tip = (
                f"Eye care: only {rate:.0f} blinks/min (relaxed is about 15-20). "
                "Blink slowly a few times so your eyes don't dry out."
            )
        else:
            tip = "Tip: keep the screen about an arm's length away."
        self._set(self.tip_label, text=tip)

    def _play_alert(self, kind):
        if not self.sound_switch.get():
            return
        if winsound is not None:
            freq, ms = {
                "distracted": (1000, 350),
                "posture": (600, 250),
                "break": (800, 200),
            }[kind]
            # Beep blocks while it plays, so give it its own tiny thread
            threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()
        else:
            self.bell()

    # -------------------------------------------------------------------------
    # CLEAN CLOSING (the X button)
    # -------------------------------------------------------------------------
    def on_close(self):
        self.stop_event.set()  # ask the worker to stop
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=3)  # wait for it to release the camera
        if self._after_id is not None:
            self.after_cancel(self._after_id)  # stop the refresh loop
        if self._upload_job is not None:
            self.after_cancel(self._upload_job)
        self.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    app = StudyBuddyApp()
    app.mainloop()
