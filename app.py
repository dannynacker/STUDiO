# app.py
# ============================================================
# RoXiva RX1 Curve Editor + Audio Scrubber
# Python Shiny implementation with progress + richer audio options
# ============================================================
#
# Install:
#   py -3.13 -m pip install shiny numpy pandas scipy matplotlib soundfile standard-aifc standard-sunau librosa
#
# Run:
#   cd "C:\Users\dn284\Desktop\roX_shiny_py"
#   py -3.13 -m shiny run --reload app.py
#
# Notes:
# - WAV is safest for first tests.
# - MP3 may work locally depending on the audio backend.
# - Use short audio files first while testing.
# - For long songs, start with 0.2-0.5 s audio step duration.

import json
import math
import tempfile
import base64
import mimetypes
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

import librosa
import librosa.display

from shiny import App, reactive, render, ui


# ============================================================
# RX1 / ROXIVA CONSTRAINTS
# ============================================================

OSC_NAMES = [f"OSC{i}" for i in range(1, 5)]
EDITOR_OSC_NAMES = OSC_NAMES + ["SUN"]
PARAM_KEYS = ["freq", "duty", "lum"]

MAX_STP_LINES = 5000

TIME_MIN_STEP = 0.1

FREQ_MIN = 0.01
FREQ_MAX = 200.00

DUTY_MIN = 1
DUTY_MAX = 99

LUM_MIN = 0
LUM_MAX = 100


# ============================================================
# MUSICAL / INTERVAL HELPERS
# ============================================================

INTERVALS = {
    "None / Unison": 0,
    "Minor Second": 1,
    "Major Second": 2,
    "Minor Third": 3,
    "Major Third": 4,
    "Perfect Fourth": 5,
    "Tritone": 6,
    "Perfect Fifth": 7,
    "Minor Sixth": 8,
    "Major Sixth": 9,
    "Minor Seventh": 10,
    "Major Seventh": 11,
    "Octave": 12,
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_hz(m):
    return 440.0 * (2.0 ** ((float(m) - 69.0) / 12.0))


def hz_to_midi(f):
    if f <= 0:
        return np.nan
    return 69.0 + 12.0 * math.log2(float(f) / 440.0)


EXTENDED_FREQS = np.array([midi_to_hz(m) for m in range(-60, 161)], dtype=float)


def hz_to_cents(f_ref, f_test):
    if f_ref <= 0 or f_test <= 0:
        return float("inf")
    return 1200.0 * math.log2(float(f_test) / float(f_ref))


def octave_candidates(freq, max_pow=14):
    if freq <= 0:
        return []
    f = float(freq)
    return [f * (2.0**k) for k in range(-max_pow, max_pow + 1)]


def octave_fold_to_range(freq, fmin, fmax):
    if freq <= 0:
        return 0.0
    f = float(freq)
    for _ in range(64):
        if f < fmin:
            f *= 2.0
        elif f > fmax:
            f /= 2.0
        else:
            break
    return float(min(max(f, fmin), fmax))


def choose_octave_near_center(freq, fmin=4.12, fmax=30.87, center_hz=11.56):
    if freq <= 0:
        return 0.0
    cands = [x for x in octave_candidates(freq) if fmin <= x <= fmax]
    if not cands:
        return octave_fold_to_range(freq, fmin, fmax)
    return float(sorted(cands, key=lambda x: abs(x - center_hz))[0])


def snap_to_musical_grid(freq, fmin=4.12, fmax=30.87, max_cents=35.0):
    if freq <= 0:
        return 0.0
    sub = EXTENDED_FREQS[(EXTENDED_FREQS >= fmin) & (EXTENDED_FREQS <= fmax)]
    if len(sub) == 0:
        return float(freq)
    idx = int(np.argmin(np.abs(sub - float(freq))))
    f0 = float(sub[idx])
    if abs(hz_to_cents(f0, float(freq))) <= max_cents:
        return f0
    return float(freq)


def apply_frequency_mapping(freq, mapping_mode, snap_music=False):
    if freq <= 0:
        return 0.0

    f = float(freq)

    if mapping_mode == "none":
        f = min(max(f, FREQ_MIN), FREQ_MAX)

    elif mapping_mode == "sls_fold":
        f = octave_fold_to_range(f, 4.12, 30.87)

    elif mapping_mode == "sls_center":
        f = choose_octave_near_center(f, 4.12, 30.87, center_hz=11.56)

    elif mapping_mode == "alpha_array":
        f = choose_octave_near_center(f, 8.18, 15.43, center_hz=11.56)

    elif mapping_mode == "rx1_safe":
        f = octave_fold_to_range(f, FREQ_MIN, FREQ_MAX)

    if snap_music and f > 0:
        f = snap_to_musical_grid(f, 4.12, 30.87, max_cents=35.0)

    return float(min(max(f, FREQ_MIN), FREQ_MAX))


def apply_interval(freq, interval_name, direction, mapping_mode, snap_music=False):
    if freq <= 0:
        return 0.0

    semitones = INTERVALS.get(interval_name, 0)
    if direction == "down":
        semitones = -semitones

    ratio = 2.0 ** (semitones / 12.0)
    f = float(freq) * ratio

    return apply_frequency_mapping(f, mapping_mode, snap_music=snap_music)


# ============================================================
# DISPLAY TRANSFORMS
# ============================================================

def freq_to_display(freq):
    freq = np.maximum(FREQ_MIN, np.minimum(FREQ_MAX, np.asarray(freq, dtype=float)))
    d = np.zeros_like(freq, dtype=float)

    for i, f in np.ndenumerate(freq):
        if f <= 5:
            d[i] = (f - FREQ_MIN) / (5 - FREQ_MIN) * 10
        elif f <= 8:
            d[i] = 10 + (f - 5) / (8 - 5) * 12
        elif f <= 15:
            d[i] = 22 + (f - 8) / (15 - 8) * 48
        elif f <= 20:
            d[i] = 70 + (f - 15) / (20 - 15) * 12
        elif f <= 30:
            d[i] = 82 + (f - 20) / (30 - 20) * 8
        elif f <= 60:
            d[i] = 90 + (f - 30) / (60 - 30) * 7
        else:
            d[i] = 97 + (f - 60) / (200 - 60) * 3

    return d


def display_to_freq(display_value):
    d = max(0.0, min(100.0, float(display_value)))

    if d <= 10:
        f = FREQ_MIN + d / 10 * (5 - FREQ_MIN)
    elif d <= 22:
        f = 5 + (d - 10) / 12 * (8 - 5)
    elif d <= 70:
        f = 8 + (d - 22) / 48 * (15 - 8)
    elif d <= 82:
        f = 15 + (d - 70) / 12 * (20 - 15)
    elif d <= 90:
        f = 20 + (d - 82) / 8 * (30 - 20)
    elif d <= 97:
        f = 30 + (d - 90) / 7 * (60 - 30)
    else:
        f = 60 + (d - 97) / 3 * (200 - 60)

    return round(max(FREQ_MIN, min(FREQ_MAX, f)), 2)


def freq_display_window(mode):
    if mode == "alpha":
        return {
            "label": "Alpha Focus: 5-20 Hz",
            "min_hz": 5,
            "max_hz": 20,
            "min_d": float(freq_to_display([5])[0]),
            "max_d": float(freq_to_display([20])[0]),
            "ticks": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20],
        }

    if mode == "low":
        return {
            "label": "Low Range: 0.01-30 Hz",
            "min_hz": 0.01,
            "max_hz": 30,
            "min_d": float(freq_to_display([0.01])[0]),
            "max_d": float(freq_to_display([30])[0]),
            "ticks": [0.01, 2, 5, 8, 10, 12, 15, 18, 20, 25, 30],
        }

    if mode == "full":
        return {
            "label": "Full RX1: 0.01-200 Hz",
            "min_hz": 0.01,
            "max_hz": 200,
            "min_d": float(freq_to_display([0.01])[0]),
            "max_d": float(freq_to_display([200])[0]),
            "ticks": [0.01, 2, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 100, 200],
        }

    return {
        "label": "SLS Design: 5-60 Hz",
        "min_hz": 5,
        "max_hz": 60,
        "min_d": float(freq_to_display([5])[0]),
        "max_d": float(freq_to_display([60])[0]),
        "ticks": [5, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 25, 30, 40, 50, 60],
    }


# ============================================================
# CURVE / RX1 HELPERS
# ============================================================

def recommended_snap_for_duration(total_duration):
    snap = math.ceil(float(total_duration) / 300.0) * 0.1
    return round(max(0.1, snap), 1)


def snap_to_grid(x, grid=0.1):
    return round(float(x) / float(grid)) * float(grid)


def clamp_time(t, total_duration, snap=0.1):
    try:
        t = float(t)
    except Exception:
        t = 0.0

    snap = max(TIME_MIN_STEP, float(snap))
    t = snap_to_grid(t, snap)
    t = max(0.0, min(float(total_duration), t))
    return round(t, 1)


def clamp_value(param, value):
    try:
        value = float(value)
    except Exception:
        value = 0.0

    if param == "freq":
        return round(max(FREQ_MIN, min(FREQ_MAX, value)), 2)

    if param == "duty":
        return int(round(max(DUTY_MIN, min(DUTY_MAX, value))))

    if param == "lum":
        return int(round(max(LUM_MIN, min(LUM_MAX, value))))

    return value


def sanitize_filename_stem(name, max_len=18):
    """
    Make a safe short filename stem for RX1/Lucia exports.

    Keeps lowercase letters and numbers only.
    Ensures the result does not start with a digit.
    """
    import re
    from pathlib import Path

    stem = Path(str(name)).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "", stem)

    if not stem:
        stem = "lucio"

    if stem[0].isdigit():
        stem = "l" + stem

    return stem[:max_len]


def renumber_points(points):
    out = points.sort_values(["osc", "param", "t"]).reset_index(drop=True)
    out["point_id"] = np.arange(1, len(out) + 1)
    return out


def clean_points_for_rx1(points, total_duration, snap=0.1):
    pts = points.copy()

    pts["t"] = [clamp_time(t, total_duration, snap) for t in pts["t"]]
    pts["value"] = [clamp_value(p, v) for p, v in zip(pts["param"], pts["value"])]

    pts = (
        pts.sort_values(["osc", "param", "t"])
        .drop_duplicates(["osc", "param", "t"], keep="last")
        .reset_index(drop=True)
    )

    return renumber_points(pts)


def value_at_time(points, osc, param, t):
    curve = (
        points[(points["osc"] == osc) & (points["param"] == param)]
        .sort_values("t")
        .reset_index(drop=True)
    )

    if len(curve) == 0:
        if param == "freq":
            return 60.0
        if param == "duty":
            return 50.0
        return 0.0

    return float(np.interp(float(t), curve["t"].values, curve["value"].values))


def ensure_boundary_points(points, total_duration, snap=0.1):
    """
    Ensures every OSC1-OSC4 freq/duty/lum curve has start/end points.

    Non-RX1 editor curves, currently used for LUCiO SUN/halogen as
    osc="SUN", param="lum", are preserved and also given boundary points.
    """
    rows = []
    points = points.copy()

    for osc in OSC_NAMES:
        for param in PARAM_KEYS:
            curve = (
                points[(points["osc"] == osc) & (points["param"] == param)]
                .sort_values("t")
                .copy()
            )

            if len(curve) == 0:
                default = 60 if param == "freq" else 50 if param == "duty" else 0
                curve = pd.DataFrame(
                    {
                        "osc": [osc, osc],
                        "param": [param, param],
                        "t": [0, total_duration],
                        "value": [default, default],
                    }
                )

            curve["t"] = curve["t"].clip(0, total_duration)
            curve = curve.drop_duplicates("t", keep="last").sort_values("t")

            if not np.any(np.isclose(curve["t"].values, 0.0)):
                v0 = float(np.interp(0.0, curve["t"].values, curve["value"].values))
                curve = pd.concat(
                    [pd.DataFrame({"osc": [osc], "param": [param], "t": [0.0], "value": [v0]}), curve],
                    ignore_index=True,
                )

            if not np.any(np.isclose(curve["t"].values, float(total_duration))):
                vend = float(np.interp(float(total_duration), curve["t"].values, curve["value"].values))
                curve = pd.concat(
                    [
                        curve,
                        pd.DataFrame({"osc": [osc], "param": [param], "t": [float(total_duration)], "value": [vend]}),
                    ],
                    ignore_index=True,
                )

            rows.append(curve[["osc", "param", "t", "value"]])

    # Preserve and boundary-pad special editor curves such as SUN/halogen.
    special = points[~points["osc"].isin(OSC_NAMES)].copy()
    if len(special) > 0:
        for (osc, param), curve in special.groupby(["osc", "param"]):
            curve = curve.sort_values("t").copy()
            curve["t"] = curve["t"].clip(0, total_duration)
            curve["value"] = [clamp_value(param, v) for v in curve["value"]]
            curve = curve.drop_duplicates("t", keep="last").sort_values("t")

            if len(curve) == 0:
                default = 0
                curve = pd.DataFrame(
                    {"osc": [osc, osc], "param": [param, param], "t": [0.0, float(total_duration)], "value": [default, default]}
                )

            if not np.any(np.isclose(curve["t"].values, 0.0)):
                v0 = float(np.interp(0.0, curve["t"].values, curve["value"].values))
                curve = pd.concat(
                    [pd.DataFrame({"osc": [osc], "param": [param], "t": [0.0], "value": [v0]}), curve],
                    ignore_index=True,
                )

            if not np.any(np.isclose(curve["t"].values, float(total_duration))):
                vend = float(np.interp(float(total_duration), curve["t"].values, curve["value"].values))
                curve = pd.concat(
                    [curve, pd.DataFrame({"osc": [osc], "param": [param], "t": [float(total_duration)], "value": [vend]})],
                    ignore_index=True,
                )

            rows.append(curve[["osc", "param", "t", "value"]])

    return clean_points_for_rx1(pd.concat(rows, ignore_index=True), total_duration, snap)

def make_default_points(total_duration=60):
    rows = []

    for osc in OSC_NAMES:
        rows.append(
            pd.DataFrame(
                {
                    "osc": osc,
                    "param": "freq",
                    "t": [0, 5, 10, 30, 50, total_duration],
                    "value": [60, 60, 10, 12, 60, 60],
                }
            )
        )

        rows.append(
            pd.DataFrame(
                {
                    "osc": osc,
                    "param": "duty",
                    "t": [0, total_duration],
                    "value": [50, 50],
                }
            )
        )

        rows.append(
            pd.DataFrame(
                {
                    "osc": osc,
                    "param": "lum",
                    "t": [0, 5, 10, 30, 50, total_duration],
                    "value": [0, 30, 50, 50, 20, 0],
                }
            )
        )

    # Optional LUCiO SUN/halogen curve. Select OSC = SUN and Parameter = Luminance to edit it.
    rows.append(
        pd.DataFrame(
            {
                "osc": "SUN",
                "param": "lum",
                "t": [0, total_duration],
                "value": [0, 0],
            }
        )
    )

    return clean_points_for_rx1(pd.concat(rows, ignore_index=True), total_duration, snap=0.1)


def selected_curve(points, osc, param):
    return points[(points["osc"] == osc) & (points["param"] == param)].sort_values("t")


def format_rx1_time(total_seconds):
    total_seconds = float(total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    fraction = int((total_seconds - int(total_seconds)) * 10)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{fraction:d}"


def led_assignments():
    return {
        "OSC1": [1, 0, 0, 0],
        "OSC2": [0, 1, 0, 0],
        "OSC3": [0, 0, 1, 0],
        "OSC4": [0, 0, 0, 1],
    }


def estimate_stp_line_count(points, total_duration, snap=0.1):
    points = ensure_boundary_points(points, total_duration, snap)
    all_times = np.unique(np.round(np.concatenate([[0], points["t"].values, [total_duration]]), 1))
    all_times = all_times[(all_times >= 0) & (all_times <= total_duration)]
    return max(0, len(all_times) - 1) + 2


def make_stp_lines(points, total_duration, snap=0.1, interpolate_between_points=False):
    points = ensure_boundary_points(points, total_duration, snap)
    all_times = np.unique(np.round(np.concatenate([[0], points["t"].values, [total_duration]]), 1))
    all_times = np.sort(all_times[(all_times >= 0) & (all_times <= total_duration)])

    line_count = max(0, len(all_times) - 1) + 2

    if line_count > MAX_STP_LINES:
        raise ValueError(f"Export would create {line_count} lines, exceeding RX1 limit of {MAX_STP_LINES}.")

    leds = led_assignments()

    lines = [
        f'TIM"{format_rx1_time(total_duration)}"',
        f'DUR"{float(total_duration):.1f}"',
    ]

    for i in range(len(all_times) - 1):
        t0 = float(all_times[i])
        t1 = float(all_times[i + 1])
        dur = round(t1 - t0, 1)

        if dur < 0.1:
            continue

        is_hold_step = (dur < 0.2) or (not bool(interpolate_between_points))
        sample_time = round((t0 + t1) / 2.0, 3)

        osc_blocks = []

        for osc in OSC_NAMES:
            if is_hold_step:
                freq_mid = clamp_value("freq", value_at_time(points, osc, "freq", sample_time))
                duty_mid = clamp_value("duty", value_at_time(points, osc, "duty", sample_time))
                lum_mid = clamp_value("lum", value_at_time(points, osc, "lum", sample_time))

                freq0 = freq_mid
                freq1 = freq_mid
                duty0 = duty_mid
                duty1 = duty_mid
                lum0 = lum_mid
                lum1 = lum_mid

            else:
                freq0 = clamp_value("freq", value_at_time(points, osc, "freq", t0))
                freq1 = clamp_value("freq", value_at_time(points, osc, "freq", t1))

                duty0 = clamp_value("duty", value_at_time(points, osc, "duty", t0))
                duty1 = clamp_value("duty", value_at_time(points, osc, "duty", t1))

                lum0 = clamp_value("lum", value_at_time(points, osc, "lum", t0))
                lum1 = clamp_value("lum", value_at_time(points, osc, "lum", t1))

            led = leds[osc]
            wave_type = 1

            block = (
                f"{wave_type},{freq0:.2f},{freq1:.2f},"
                f"{int(duty0)},{int(duty1)},"
                f"{led[0]},{led[1]},{led[2]},{led[3]},"
                f"{int(lum0)},{int(lum1)}"
            )
            osc_blocks.append(block)

        lines.append(f'STP"{dur:.1f},{",".join(osc_blocks)}"')

    if len(lines) > MAX_STP_LINES:
        raise ValueError(f"Export created {len(lines)} lines, exceeding RX1 limit of {MAX_STP_LINES}.")

    return lines


# ============================================================
# LUCiO / LUCIA .LSCF EXPORT HELPERS
# ============================================================

LUCIO_HEADER_LEN = 132
LUCIO_ROW_LEN = 56
LUCIO_CHECKSUM_LEN = 1
LUCIO_NAME_FIELD_OFFSET = 24
LUCIO_NAME_FIELD_LEN = 56
LUCIO_DUTY_OFFSETS = [3, 11, 19, 27]

LUCIO_FREQ_MIN = 0.0
LUCIO_FREQ_MAX = 60.0
LUCIO_DUTY_MIN = 10
LUCIO_DUTY_MAX = 90
LUCIO_LUM_MIN = 0
LUCIO_LUM_MAX = 100
LUCIO_SUN_MIN = 0
LUCIO_SUN_MAX = 100


def lucio_xor_checksum(data_without_checksum: bytes) -> int:
    checksum = 0
    for byte in data_without_checksum:
        checksum ^= byte
    return checksum


def lucio_finalize_file(buffer: bytearray) -> bytearray:
    buffer = bytearray(buffer)
    buffer[-1] = lucio_xor_checksum(buffer[:-1])
    return buffer


def lucio_verify_checksum(data: bytes) -> bool:
    return data[-1] == lucio_xor_checksum(data[:-1])


def lucio_read_internal_name(data: bytes) -> str:
    raw = bytes(data[LUCIO_NAME_FIELD_OFFSET:LUCIO_NAME_FIELD_OFFSET + LUCIO_NAME_FIELD_LEN])
    raw = raw.split(b"\x00", 1)[0]
    return raw.decode("ascii", errors="replace")


def lucio_parse_rows(data: bytes) -> list[bytearray]:
    n_rows = (len(data) - LUCIO_HEADER_LEN - LUCIO_CHECKSUM_LEN) // LUCIO_ROW_LEN
    expected_len = LUCIO_HEADER_LEN + n_rows * LUCIO_ROW_LEN + LUCIO_CHECKSUM_LEN

    if len(data) != expected_len:
        raise ValueError(
            f"Invalid .lscf size: {len(data)} bytes. Expected 132 + 56*N + 1 bytes."
        )

    return [
        bytearray(data[LUCIO_HEADER_LEN + i * LUCIO_ROW_LEN:LUCIO_HEADER_LEN + (i + 1) * LUCIO_ROW_LEN])
        for i in range(n_rows)
    ]


def lucio_main_cycle_seconds_from_row(row: bytes) -> float:
    stored = struct.unpack("<H", bytes(row[44:46]))[0]
    if stored <= 0:
        return 2.5
    return 2500.0 / float(stored)


def lucio_cycles_from_hz(freq_hz: float, main_cycle_seconds: float) -> int:
    cycles = int(round(float(freq_hz) * float(main_cycle_seconds)))
    return int(np.clip(cycles, 1, 255))


def lucio_write_halogen_bytes(row: bytearray, halogen: int) -> None:
    """
    Empirically validated LUCiO-Composer-SUN row-control patterns:
      halogen off: 00 01 00 14
      halogen on:  01 01 01 H
    """
    halogen = int(np.clip(round(halogen), LUCIO_SUN_MIN, LUCIO_SUN_MAX))
    if halogen <= 0:
        row[40:44] = bytes([0x00, 0x01, 0x00, 0x14])
    else:
        row[40:44] = bytes([0x01, 0x01, 0x01, halogen])


def lucio_sun_value_at_time(points, t):
    sun_curve = points[(points["osc"] == "SUN") & (points["param"] == "lum")].sort_values("t")
    if len(sun_curve) == 0:
        return 0
    return int(np.clip(round(np.interp(float(t), sun_curve["t"].values, sun_curve["value"].values)), 0, 100))


def lucio_build_control_table_from_points(points, total_duration, control_step_seconds=1.0):
    """
    Samples the interactive editor curves into one discrete Lucia target state per row.
    """
    total_duration = float(total_duration)
    control_step_seconds = float(control_step_seconds)
    if control_step_seconds <= 0:
        raise ValueError("LUCiO control step must be positive.")

    pts = ensure_boundary_points(points, total_duration, snap=max(0.1, control_step_seconds))
    n_rows = max(1, int(math.ceil(total_duration / control_step_seconds)))

    rows = []
    for i in range(n_rows):
        t = min(i * control_step_seconds, total_duration)
        row = {"row": i + 1, "time_sec": t, "halogen": lucio_sun_value_at_time(pts, t)}

        for osc in OSC_NAMES:
            freq = value_at_time(pts, osc, "freq", t)
            duty = value_at_time(pts, osc, "duty", t)
            lum = value_at_time(pts, osc, "lum", t)

            row[f"{osc}_freq_hz"] = float(np.clip(freq, LUCIO_FREQ_MIN, LUCIO_FREQ_MAX))
            row[f"{osc}_duty"] = int(np.clip(round(duty), LUCIO_DUTY_MIN, LUCIO_DUTY_MAX))
            row[f"{osc}_lum"] = int(np.clip(round(lum), LUCIO_LUM_MIN, LUCIO_LUM_MAX))

        rows.append(row)

    return pd.DataFrame(rows)


def clamp_byte(value):
    """
    Clamp a numeric value to a valid unsigned byte, 0-255.
    Used when writing Lucia row bytes.
    """
    try:
        value = float(value)
    except Exception:
        value = 0.0

    return int(np.clip(round(value), 0, 255))


def lucio_patch_row(template_row: bytes, control_row: pd.Series, displayed_seconds_tenths=10, loops_per_row=1):
    row = bytearray(template_row)
    main_cycle_seconds = lucio_main_cycle_seconds_from_row(row)
    halogen = int(control_row["halogen"])

    debug = {
        "main_cycle_seconds": main_cycle_seconds,
        "displayed_row_seconds": displayed_seconds_tenths / 10.0,
        "halogen": halogen,
        "row_40_44_hex": None,
        "row_40_56_hex": None,
    }

    for osc_idx, osc in enumerate(OSC_NAMES):
        base = osc_idx * 8

        freq_hz = float(control_row[f"{osc}_freq_hz"])
        duty = int(control_row[f"{osc}_duty"])
        lum = int(control_row[f"{osc}_lum"])
        cycles = lucio_cycles_from_hz(freq_hz, main_cycle_seconds)

        active_flag = 1

        row[base + 0] = active_flag
        row[base + 1] = active_flag
        row[base + 2] = 100
        row[base + 3] = clamp_byte(duty)
        row[base + 4] = clamp_byte(cycles)
        row[base + 5] = 0
        row[base + 6] = active_flag
        row[base + 7] = 0

        row[32 + osc_idx] = clamp_byte(lum)
        row[36 + osc_idx] = active_flag

        debug[f"{osc}_cycles"] = cycles
        debug[f"{osc}_achieved_freq_hz"] = cycles / main_cycle_seconds

    lucio_write_halogen_bytes(row, halogen)

    row[46] = 1
    row[47] = 1
    row[48] = clamp_byte(displayed_seconds_tenths)
    row[49] = 0
    row[50] = 0
    row[51] = 0
    row[52:56] = struct.pack("<I", int(loops_per_row))

    debug["row_40_44_hex"] = row[40:44].hex(" ")
    debug["row_40_56_hex"] = row[40:56].hex(" ")

    return row, debug


def lucio_build_lscf_from_points(points, total_duration, template_data, control_step_seconds=1.0):
    template_data = bytearray(template_data)
    if not lucio_verify_checksum(template_data):
        raise ValueError("Template checksum is invalid. Upload the validated LUCiO dynamic-duty template.")

    template_rows = lucio_parse_rows(template_data)
    template_row = template_rows[0]
    control_df = lucio_build_control_table_from_points(points, total_duration, control_step_seconds)

    displayed_seconds_tenths = int(round(control_step_seconds * 10))
    displayed_seconds_tenths = int(np.clip(displayed_seconds_tenths, 1, 255))

    out = bytearray()
    out.extend(template_data[:LUCIO_HEADER_LEN])

    debug_rows = []
    for _, control_row in control_df.iterrows():
        patched_row, row_debug = lucio_patch_row(
            template_row,
            control_row,
            displayed_seconds_tenths=displayed_seconds_tenths,
            loops_per_row=1,
        )
        out.extend(patched_row)

        debug_row = dict(control_row)
        debug_row.update(row_debug)
        debug_rows.append(debug_row)

    out.extend(b"\x00")
    out = lucio_finalize_file(out)

    if not lucio_verify_checksum(out):
        raise RuntimeError("Final LUCiO XOR checksum failed.")

    debug_df = pd.DataFrame(debug_rows)
    return bytes(out), debug_df


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def dedupe_frequency_candidates(freqs, amps, cents_sep=35.0):
    pairs = sorted(zip(freqs, amps), key=lambda x: -x[1])
    kept_f = []
    kept_a = []

    for f, a in pairs:
        if f <= 0 or a <= 0:
            continue

        ok = True
        for fk in kept_f:
            if abs(hz_to_cents(fk, f)) < cents_sep:
                ok = False
                break

        if ok:
            kept_f.append(float(f))
            kept_a.append(float(a))

    return kept_f, kept_a


def harmonic_salience_score(freqs_band, spec, peak_freqs):
    scores = []

    harmonics = [1, 2, 3, 4, 5]
    weights = [1.0, 0.60, 0.40, 0.25, 0.18]

    for f0 in peak_freqs:
        score = 0.0
        for h, w in zip(harmonics, weights):
            ft = float(f0) * h
            if ft < freqs_band[0] or ft > freqs_band[-1]:
                continue
            j = int(np.argmin(np.abs(freqs_band - ft)))
            score += w * float(spec[j])

        score *= 1.0 + 0.18 * (float(freqs_band[-1]) / max(float(f0), 1.0)) ** 0.15
        scores.append(score)

    return np.asarray(scores, dtype=float)


def cents_mask(freqs, centre_hz, bw_cents):
    if centre_hz <= 0:
        return np.zeros_like(freqs, dtype=bool)
    cents = 1200.0 * np.log2(np.maximum(freqs, 1e-9) / float(centre_hz))
    return np.abs(cents) <= float(bw_cents)


def local_band_energy_ratio(freqs_band, spec, centre_hz, bw_cents=35.0):
    if centre_hz <= 0:
        return 0.0

    total = float(np.sum(spec))
    if total <= 0:
        return 0.0

    mask = cents_mask(freqs_band, centre_hz, bw_cents)
    if not np.any(mask):
        return 0.0

    return float(np.sum(spec[mask]) / total)


def harmonic_band_energy_ratio(freqs_band, spec, centre_hz, bw_cents=35.0):
    if centre_hz <= 0:
        return 0.0

    total = float(np.sum(spec))
    if total <= 0:
        return 0.0

    harmonics = [1, 2, 3, 4, 5]
    weights = [1.0, 0.60, 0.40, 0.25, 0.18]

    num = 0.0
    den_w = 0.0

    for h, w in zip(harmonics, weights):
        ft = float(centre_hz) * h
        if ft < freqs_band[0] or ft > freqs_band[-1]:
            continue
        mask = cents_mask(freqs_band, ft, bw_cents)
        if np.any(mask):
            num += float(w) * float(np.sum(spec[mask]))
            den_w += float(w)

    if den_w <= 0:
        return 0.0

    return float(num / total)


def normalize_amplitudes(amps_by_voice, method="global_linear", percentile=95.0):
    arr = np.asarray(amps_by_voice, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0

    if arr.size == 0:
        return arr.astype(int)

    if method == "per_voice_linear":
        denom = np.max(arr, axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        out = arr / denom * 100.0

    elif method == "percentile_clip":
        denom = float(np.percentile(arr[arr > 0], percentile)) if np.any(arr > 0) else 1.0
        denom = max(denom, 1e-9)
        out = np.clip(arr / denom * 100.0, 0, 100)

    elif method == "log_global":
        denom = float(np.max(arr))
        denom = max(denom, 1e-9)
        out = np.log1p(arr) / np.log1p(denom) * 100.0

    elif method == "sqrt_global":
        denom = float(np.max(arr))
        denom = max(denom, 1e-9)
        out = np.sqrt(arr / denom) * 100.0

    else:
        denom = float(np.max(arr))
        denom = max(denom, 1e-9)
        out = arr / denom * 100.0

    return np.clip(np.round(out), 0, 100).astype(int)


def smooth_series(values, method="none", window=1):
    x = pd.Series(values)
    window = int(max(1, window))

    if method == "rolling_mean" and window > 1:
        return x.rolling(window=window, center=True, min_periods=1).mean().values

    if method == "rolling_median" and window > 1:
        return x.rolling(window=window, center=True, min_periods=1).median().values

    return np.asarray(values)


def make_duty_from_options(
    amp_pct,
    occupancy_pct,
    harmonic_pct,
    duty_method="fixed",
    fixed_duty=50,
    duty_floor=10,
    duty_ceiling=90,
    amp_gate=0,
):
    amp_pct = np.asarray(amp_pct, dtype=float)
    occupancy_pct = np.asarray(occupancy_pct, dtype=float)
    harmonic_pct = np.asarray(harmonic_pct, dtype=float)

    duty_floor = float(duty_floor)
    duty_ceiling = float(duty_ceiling)
    amp_gate = float(amp_gate)

    if duty_method == "amplitude":
        duty = amp_pct

    elif duty_method == "inverted_amplitude":
        duty = 100.0 - amp_pct

    elif duty_method == "wide_at_high_amp":
        duty = duty_floor + (amp_pct / 100.0) * (duty_ceiling - duty_floor)

    elif duty_method == "narrow_at_high_amp":
        duty = duty_ceiling - (amp_pct / 100.0) * (duty_ceiling - duty_floor)

    elif duty_method == "occupancy_proxy":
        duty = duty_floor + (occupancy_pct / 100.0) * (duty_ceiling - duty_floor)

    elif duty_method == "harmonic_band_proxy":
        duty = duty_floor + (harmonic_pct / 100.0) * (duty_ceiling - duty_floor)

    elif duty_method == "amp_gated_occupancy":
        duty = duty_floor + (occupancy_pct / 100.0) * (duty_ceiling - duty_floor)
        duty = np.where(amp_pct >= amp_gate, duty, duty_floor)

    else:
        duty = np.repeat(float(fixed_duty), len(amp_pct))

    return np.clip(np.round(duty), DUTY_MIN, DUTY_MAX).astype(int)


def make_luminance_from_options(
    amp_pct,
    lum_method="amplitude",
    fixed_lum=50,
    lum_floor=0,
    lum_ceiling=100,
    amp_gate=0,
):
    amp_pct = np.asarray(amp_pct, dtype=float)

    lum_floor = float(lum_floor)
    lum_ceiling = float(lum_ceiling)
    amp_gate = float(amp_gate)

    if lum_method == "fixed":
        lum = np.repeat(float(fixed_lum), len(amp_pct))

    elif lum_method == "sqrt_amplitude":
        lum = lum_floor + np.sqrt(amp_pct / 100.0) * (lum_ceiling - lum_floor)

    elif lum_method == "log_amplitude":
        lum = lum_floor + (np.log1p(amp_pct) / np.log1p(100.0)) * (lum_ceiling - lum_floor)

    elif lum_method == "threshold_gated":
        lum = lum_floor + (amp_pct / 100.0) * (lum_ceiling - lum_floor)
        lum = np.where(amp_pct >= amp_gate, lum, 0)

    else:
        lum = lum_floor + (amp_pct / 100.0) * (lum_ceiling - lum_floor)

    return np.clip(np.round(lum), LUM_MIN, LUM_MAX).astype(int)


def analyze_audio_file(
    audio_path,
    engine="fft",
    n_voices=1,
    step_duration=0.1,
    n_fft=4096,
    band_lo=40.0,
    band_hi=6000.0,
    peak_rel_height=0.20,
    mapping_mode="sls_center",
    snap_music=False,
    interval_name="None / Unison",
    interval_direction="up",
    amplitude_norm="global_linear",
    amp_percentile=95.0,
    duty_method="fixed",
    fixed_duty=50,
    duty_floor=10,
    duty_ceiling=90,
    duty_amp_gate=0,
    luminance_method="amplitude",
    fixed_lum=50,
    lum_floor=0,
    lum_ceiling=100,
    lum_amp_gate=0,
    occupancy_bw_cents=35.0,
    smoothing_method="none",
    smoothing_window=1,
    progress=None,
):
    def tick(value, message, detail=""):
        if progress is not None:
            progress(value, message, detail)

    tick(0.01, "Stage 1/6: loading audio", "Reading file with librosa")

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    if y.size == 0:
        raise ValueError("Could not read audio, or audio file is empty.")

    audio_duration = len(y) / float(sr)

    hop_length = max(1, int(round(sr * float(step_duration))))
    n_voices = int(max(1, min(4, n_voices)))

    tick(0.08, "Stage 2/6: computing spectrogram", f"Duration {audio_duration:.1f} s, sample rate {sr} Hz")

    if engine == "cqt":
        fmin = librosa.note_to_hz("C1")
        n_bins = 84
        bins_per_octave = 12

        spectrum_matrix = np.abs(
            librosa.cqt(
                y.astype(np.float32),
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
            )
        )

        freqs_all = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
        frames = spectrum_matrix.shape[1]
        times = librosa.frames_to_time(np.arange(frames), sr=sr, hop_length=hop_length)

    else:
        spectrum_matrix = np.abs(
            librosa.stft(
                y.astype(np.float32),
                n_fft=int(n_fft),
                hop_length=hop_length,
                window="hann",
                center=True,
            )
        )

        freqs_all = librosa.fft_frequencies(sr=sr, n_fft=int(n_fft))
        frames = spectrum_matrix.shape[1]
        times = librosa.frames_to_time(np.arange(frames), sr=sr, hop_length=hop_length)

    band_mask = (freqs_all >= float(band_lo)) & (freqs_all <= float(band_hi))
    band_bins = np.flatnonzero(band_mask)

    if len(band_bins) == 0:
        raise ValueError("No spectral bins in selected frequency band.")

    freqs_band = freqs_all[band_bins]

    raw_freqs_by_voice = [[] for _ in range(4)]
    mapped_freqs_by_voice = [[] for _ in range(4)]
    amps_by_voice = [[] for _ in range(4)]
    occupancy_by_voice = [[] for _ in range(4)]
    harmonic_by_voice = [[] for _ in range(4)]

    tick(0.18, "Stage 3/6: extracting spectral peaks", f"{frames} frames to process")

    update_every = max(1, frames // 100)

    for frame_idx in range(frames):
        spec = spectrum_matrix[band_bins, frame_idx]

        if spec.size == 0 or float(np.max(spec)) <= 0:
            selected_f = [0.0] * 4
            selected_a = [0.0] * 4
            selected_occ = [0.0] * 4
            selected_harm = [0.0] * 4

        else:
            threshold = float(np.max(spec)) * float(peak_rel_height)
            peaks, _ = find_peaks(spec, height=threshold)

            if len(peaks) == 0:
                peaks = np.array([int(np.argmax(spec))], dtype=int)

            peak_freqs = freqs_band[peaks]
            peak_amps = spec[peaks]

            if engine == "harmonic_fft":
                sort_scores = harmonic_salience_score(freqs_band, spec, peak_freqs)
            else:
                sort_scores = peak_amps

            order = np.argsort(-sort_scores)

            cand_freqs = [float(peak_freqs[i]) for i in order]
            cand_amps = [float(sort_scores[i]) for i in order]

            dedup_f, dedup_a = dedupe_frequency_candidates(cand_freqs, cand_amps, cents_sep=35.0)

            selected_f = []
            selected_a = []
            selected_occ = []
            selected_harm = []

            for f, a in zip(dedup_f, dedup_a):
                selected_f.append(float(f))
                selected_a.append(float(a))
                selected_occ.append(local_band_energy_ratio(freqs_band, spec, f, bw_cents=occupancy_bw_cents))
                selected_harm.append(harmonic_band_energy_ratio(freqs_band, spec, f, bw_cents=occupancy_bw_cents))

                if len(selected_f) >= n_voices:
                    break

            while len(selected_f) < 4:
                selected_f.append(0.0)
                selected_a.append(0.0)
                selected_occ.append(0.0)
                selected_harm.append(0.0)

        for vi in range(4):
            raw_f = selected_f[vi] if vi < n_voices else 0.0
            raw_a = selected_a[vi] if vi < n_voices else 0.0
            occ = selected_occ[vi] if vi < n_voices else 0.0
            harm = selected_harm[vi] if vi < n_voices else 0.0

            mapped_f = apply_frequency_mapping(raw_f, mapping_mode, snap_music=snap_music)
            mapped_f = apply_interval(mapped_f, interval_name, interval_direction, mapping_mode, snap_music=snap_music)

            raw_freqs_by_voice[vi].append(float(raw_f))
            mapped_freqs_by_voice[vi].append(float(mapped_f))
            amps_by_voice[vi].append(float(raw_a))
            occupancy_by_voice[vi].append(float(occ))
            harmonic_by_voice[vi].append(float(harm))

        if frame_idx % update_every == 0 or frame_idx == frames - 1:
            prop = frame_idx / max(1, frames - 1)
            tick(
                0.18 + prop * 0.50,
                "Stage 3/6: extracting spectral peaks",
                f"Frame {frame_idx + 1} of {frames}",
            )

    tick(0.70, "Stage 4/6: normalising amplitude", "Computing RX1 amplitude/luminance scale")

    amp_pct_all = normalize_amplitudes(
        amps_by_voice,
        method=amplitude_norm,
        percentile=float(amp_percentile),
    )

    occupancy_pct_all = np.clip(np.round(np.asarray(occupancy_by_voice) * 100.0), 0, 100).astype(int)
    harmonic_pct_all = np.clip(np.round(np.asarray(harmonic_by_voice) * 100.0), 0, 100).astype(int)

    tick(0.78, "Stage 5/6: computing duty/luminance", "Applying selected strobification options")

    out = pd.DataFrame({"Time": np.asarray(times, dtype=float)})

    for vi in range(4):
        ch = vi + 1

        raw_freq = np.asarray(raw_freqs_by_voice[vi], dtype=float)
        mapped_freq = np.asarray(mapped_freqs_by_voice[vi], dtype=float)
        amp_pct = amp_pct_all[vi]
        occ_pct = occupancy_pct_all[vi]
        harm_pct = harmonic_pct_all[vi]

        duty = make_duty_from_options(
            amp_pct=amp_pct,
            occupancy_pct=occ_pct,
            harmonic_pct=harm_pct,
            duty_method=duty_method,
            fixed_duty=fixed_duty,
            duty_floor=duty_floor,
            duty_ceiling=duty_ceiling,
            amp_gate=duty_amp_gate,
        )

        lum = make_luminance_from_options(
            amp_pct=amp_pct,
            lum_method=luminance_method,
            fixed_lum=fixed_lum,
            lum_floor=lum_floor,
            lum_ceiling=lum_ceiling,
            amp_gate=lum_amp_gate,
        )

        if smoothing_method != "none" and int(smoothing_window) > 1:
            mapped_freq = smooth_series(mapped_freq, smoothing_method, smoothing_window)
            amp_pct = smooth_series(amp_pct, smoothing_method, smoothing_window)
            occ_pct = smooth_series(occ_pct, smoothing_method, smoothing_window)
            harm_pct = smooth_series(harm_pct, smoothing_method, smoothing_window)
            duty = smooth_series(duty, smoothing_method, smoothing_window)
            lum = smooth_series(lum, smoothing_method, smoothing_window)

        out[f"Audio_RawFreq_{ch}"] = np.asarray(raw_freq, dtype=float)
        out[f"Audio_Freq_{ch}"] = np.round(np.asarray(mapped_freq, dtype=float), 2)
        out[f"Audio_Amp_{ch}"] = np.clip(np.round(amp_pct), 0, 100).astype(int)
        out[f"Audio_Occupancy_{ch}"] = np.clip(np.round(occ_pct), 0, 100).astype(int)
        out[f"Audio_HarmonicBand_{ch}"] = np.clip(np.round(harm_pct), 0, 100).astype(int)
        out[f"Audio_Duty_{ch}"] = np.clip(np.round(duty), DUTY_MIN, DUTY_MAX).astype(int)
        out[f"Audio_Lum_{ch}"] = np.clip(np.round(lum), LUM_MIN, LUM_MAX).astype(int)

    tick(0.94, "Stage 6/6: finishing", "Preparing scrubber dataframe")

    tick(1.00, "Audio Analysis complete", f"{len(out)} frames")

    return out


def audio_trace_to_points(audio_df, osc, voice, include_freq=True, include_duty=True, include_lum=True):
    rows = []
    ch = int(voice)

    for _, row in audio_df.iterrows():
        t = float(row["Time"])

        if include_freq:
            rows.append({"osc": osc, "param": "freq", "t": t, "value": float(row[f"Audio_Freq_{ch}"])})

        if include_duty:
            rows.append({"osc": osc, "param": "duty", "t": t, "value": int(row[f"Audio_Duty_{ch}"])})

        if include_lum:
            rows.append({"osc": osc, "param": "lum", "t": t, "value": int(row[f"Audio_Lum_{ch}"])})

    return pd.DataFrame(rows)

# ============================================================
# AUDIO PLAYBACK / DIAGNOSTIC VIEW HELPERS
# ============================================================

def file_to_data_uri(path, filename=None):
    """
    Encodes the uploaded audio file so the browser can play it with <audio controls>.
    This is simple and portable for short/medium files. For very long files, a static
    file-serving route would be more efficient, but this is much easier for the MVP.
    """
    path = Path(path)

    mime = None
    if filename is not None:
        mime = mimetypes.guess_type(filename)[0]

    if mime is None:
        mime = mimetypes.guess_type(str(path))[0]

    if mime is None:
        mime = "audio/wav"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    return f"data:{mime};base64,{encoded}"


def get_uploaded_audio_path(input):
    files = input.audio_file()
    if not files:
        return None, None

    path = files[0]["datapath"]
    name = files[0].get("name", "uploaded_audio")
    return path, name


def extract_playhead_time(input):
    """
    Reads browser playback time sent by JavaScript.
    """
    ph = input.audio_playhead()
    if ph is None:
        return None

    try:
        return float(ph.get("t", 0.0))
    except Exception:
        return None


def add_playhead_line(ax, playhead_time, colour="red"):
    if playhead_time is None:
        return

    try:
        t = float(playhead_time)
    except Exception:
        return

    ax.axvline(t, color=colour, linewidth=1.4, alpha=0.85)








def plot_audio_waveform_spectrogram(audio_path, playhead_time=None):
    """
    Stable Audacity-like diagnostic view.

    This renderer is intentionally display-optimised rather than analysis-optimised:
    large MP3s are downsampled for the waveform/spectrogram preview so Matplotlib
    does not try to allocate massive RGBA arrays.
    """
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.55]},
    )

    ax_wave = axes[0]
    ax_spec = axes[1]

    if audio_path is None:
        ax_wave.text(0.5, 0.5, "Upload Audio to See Waveform.", ha="center", va="center")
        ax_spec.text(0.5, 0.5, "Upload Audio to See Spectrogram.", ha="center", va="center")
        ax_wave.set_axis_off()
        ax_spec.set_axis_off()
        ax_wave.set_position([0.075, 0.63, 0.870, 0.25])
        ax_spec.set_position([0.075, 0.12, 0.870, 0.42])
        return fig

    try:
        # Diagnostic display load: lower sample rate is enough for the visual preview.
        # This keeps long MP3s from producing enormous spectrogram images.
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        y = y.astype(np.float32, copy=False)

        if y.size == 0:
            raise ValueError("Audio appears to be empty.")

        duration = len(y) / float(sr)

        # -------------------------
        # Waveform, decimated display
        # -------------------------
        max_wave_points = 12000
        wave_step = max(1, int(np.ceil(len(y) / max_wave_points)))
        y_plot = y[::wave_step]
        t_plot = np.arange(len(y_plot), dtype=np.float32) * wave_step / float(sr)

        ax_wave.plot(t_plot, y_plot, linewidth=0.35)
        ax_wave.set_xlim(0, duration)
        ax_wave.set_ylabel("Waveform", labelpad=4)
        ax_wave.set_title("Uploaded Audio: Waveform and Spectrogram")
        ax_wave.grid(True, alpha=0.20)
        ax_wave.tick_params(labelbottom=False)

        # -------------------------
        # Spectrogram, capped display size
        # -------------------------
        n_fft = 1024

        # Choose hop so long files do not create tens of thousands of columns.
        max_spec_cols = 1800
        hop_length = max(512, int(np.ceil(len(y) / max_spec_cols)))

        S = np.abs(
            librosa.stft(
                y,
                n_fft=n_fft,
                hop_length=hop_length,
                window="hann",
                center=True,
            )
        ).astype(np.float32, copy=False)

        S_db = librosa.amplitude_to_db(S, ref=np.max).astype(np.float32, copy=False)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)

        f_hi = min(sr / 2.0, 19000.0)
        f_mask = (freqs >= 50.0) & (freqs <= f_hi)

        img_data = S_db[f_mask, :]
        freqs_used = freqs[f_mask]

        # Cap frequency rows too. For display, 512 rows is plenty.
        max_rows = 512
        if img_data.shape[0] > max_rows:
            row_idx = np.linspace(0, img_data.shape[0] - 1, max_rows).astype(int)
            img_data = img_data[row_idx, :]
            freqs_used = freqs_used[row_idx]

        # Cap columns in case hop logic still leaves too many.
        if img_data.shape[1] > max_spec_cols:
            col_idx = np.linspace(0, img_data.shape[1] - 1, max_spec_cols).astype(int)
            img_data = img_data[:, col_idx]

        extent = [0, duration, float(freqs_used[0]), float(freqs_used[-1])]

        ax_spec.imshow(
            img_data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=-80,
            vmax=0,
            interpolation="nearest",
        )

        ax_spec.set_xlim(0, duration)
        ax_spec.set_ylim(float(freqs_used[0]), float(freqs_used[-1]))
        ax_spec.set_yscale("log")
        ax_spec.set_ylabel("Frequency (Hz)", labelpad=4)
        ax_spec.set_xlabel("Time (s)")

        ax_wave.set_position([0.075, 0.63, 0.870, 0.25])
        ax_spec.set_position([0.075, 0.12, 0.870, 0.42])

        return fig

    except Exception as e:
        ax_wave.text(
            0.5,
            0.5,
            "Could not render audio view:\n" + str(e),
            ha="center",
            va="center",
        )
        ax_spec.set_axis_off()
        ax_wave.set_axis_off()
        ax_wave.set_position([0.075, 0.63, 0.870, 0.25])
        ax_spec.set_position([0.075, 0.12, 0.870, 0.42])
        return fig

# ============================================================
# SVG EDITOR
# ============================================================

def make_svg_editor(
    points,
    selected_osc,
    selected_param,
    total_duration,
    snap,
    view_start,
    view_end,
    freq_view_mode,
    add_mode=False,
    audio_df=None,
    overlay_audio_freq=False,
    overlay_audio_duty=False,
    overlay_audio_lum=False,
    overlay_voice=1,
):
    points = ensure_boundary_points(points, total_duration, snap=snap)

    view_start = max(0.0, min(float(view_start), float(total_duration)))
    view_end = max(0.0, min(float(view_end), float(total_duration)))

    if view_end <= view_start:
        view_start = 0.0
        view_end = float(total_duration)

    view_width = view_end - view_start

    fwin = freq_display_window(freq_view_mode)
    freq_d_min = fwin["min_d"]
    freq_d_max = fwin["max_d"]
    freq_d_width = freq_d_max - freq_d_min

    width = 1120
    height = 470

    margin_left = 82
    margin_right = 72
    margin_top = 32
    margin_bottom = 58

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_to_px(t):
        return margin_left + ((float(t) - view_start) / view_width) * plot_w

    def freq_to_py(v):
        d = float(freq_to_display([v])[0])
        return margin_top + (1 - ((d - freq_d_min) / freq_d_width)) * plot_h

    def pct_to_py(v):
        return margin_top + (1 - float(v) / 100.0) * plot_h

    def polyline_points(curve, param):
        coords = []
        for _, row in curve.sort_values("t").iterrows():
            x = x_to_px(row["t"])
            y = freq_to_py(row["value"]) if param == "freq" else pct_to_py(row["value"])
            coords.append(f"{x:.2f},{y:.2f}")
        return " ".join(coords)

    def audio_polyline(df, col, param):
        coords = []
        use = df[(df["Time"] >= view_start) & (df["Time"] <= view_end)]
        for _, row in use.iterrows():
            x = x_to_px(row["Time"])
            y = freq_to_py(row[col]) if param == "freq" else pct_to_py(row[col])
            coords.append(f"{x:.2f},{y:.2f}")
        return " ".join(coords)

    osc_colours = {
        "OSC1": "#1f77b4",
        "OSC2": "#d62728",
        "OSC3": "#2ca02c",
        "OSC4": "#9467bd",
    }

    param_dash = {
        "freq": "",
        "duty": "8,5",
        "lum": "2,5",
    }

    html = []

    html.append(
        f'''
<svg id="curve_svg" class="curve-svg" width="100%" height="{height}"
     viewBox="0 0 {width} {height}"
     preserveAspectRatio="none"
     data-total-duration="{float(total_duration):.6f}"
     data-view-start="{view_start:.6f}"
     data-view-end="{view_end:.6f}"
     data-selected-param="{selected_param}"
     data-snap="{float(snap):.6f}"
     data-plot-left="{margin_left:.6f}"
     data-plot-right="{width - margin_right:.6f}"
     data-plot-top="{margin_top:.6f}"
     data-plot-bottom="{height - margin_bottom:.6f}"
     data-freq-display-min="{freq_d_min:.6f}"
     data-freq-display-max="{freq_d_max:.6f}"
     data-pct-min="0"
     data-pct-max="100"
     style="border:1px solid #ccc; background:white; cursor:{'crosshair' if add_mode else 'default'};">
'''
    )

    html.append(
        f'''
<defs>
  <clipPath id="plot_clip">
    <rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}"></rect>
  </clipPath>
</defs>
'''
    )

    html.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#dddddd"/>'
    )

    x_ticks = np.linspace(view_start, view_end, 9)
    for xt in x_ticks:
        x = x_to_px(xt)
        html.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{margin_top}" y2="{height - margin_bottom}" stroke="#e6e6e6"/>')
        html.append(f'<text x="{x:.2f}" y="{height - margin_bottom + 19}" font-size="11" text-anchor="middle" fill="#333">{xt:.1f}</text>')

    for yt in fwin["ticks"]:
        y = freq_to_py(yt)
        if y < margin_top - 5 or y > height - margin_bottom + 5:
            continue
        html.append(f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#eeeeee"/>')
        html.append(f'<text x="{margin_left - 6}" y="{y:.2f}" font-size="10" text-anchor="end" dominant-baseline="middle" fill="#1f4aa8">{yt:g}</text>')

    for yt in [0, 20, 40, 60, 80, 100]:
        y = pct_to_py(yt)
        html.append(f'<text x="{width - margin_right + 6}" y="{y:.2f}" font-size="10" text-anchor="start" dominant-baseline="middle" fill="#333">{yt:g}</text>')

    html.append(f'<text x="{margin_left + plot_w / 2}" y="{height - 16}" font-size="12" text-anchor="middle" fill="#333">Time (s)</text>')

    html.append(
        f'<text x="23" y="{margin_top + plot_h / 2}" font-size="12" text-anchor="middle" fill="#1f4aa8" '
        f'transform="rotate(-90,23,{margin_top + plot_h / 2})">Frequency: {fwin["label"]}</text>'
    )

    html.append(
        f'<text x="{width - 18}" y="{margin_top + plot_h / 2}" font-size="12" text-anchor="middle" fill="#333" '
        f'transform="rotate(90,{width - 18},{margin_top + plot_h / 2})">Duty Cycle & Luminance (%)</text>'
    )

    html.append('<g clip-path="url(#plot_clip)">')

    # Draw OSC1-OSC4 curves plus any special editor curves such as SUN/halogen.
    drawable = points[points["param"].isin(PARAM_KEYS)].copy()
    for (osc, param), curve in drawable.groupby(["osc", "param"]):
        curve = curve.sort_values("t")
        if len(curve) < 2:
            continue

        selected = osc == selected_osc and param == selected_param
        stroke_colour = osc_colours.get(osc, "#f2a100")

        html.append(
            f'<polyline points="{polyline_points(curve, param)}" fill="none" '
            f'stroke="{stroke_colour}" stroke-width="{3.2 if selected else 1.6}" '
            f'stroke-dasharray="{param_dash.get(param, "")}" opacity="{1.0 if selected else 0.35}" pointer-events="none"/>'
        )

    if audio_df is not None and len(audio_df) > 1:
        ch = int(overlay_voice)

        if overlay_audio_freq and f"Audio_Freq_{ch}" in audio_df.columns:
            pts = audio_polyline(audio_df, f"Audio_Freq_{ch}", "freq")
            html.append(f'<polyline points="{pts}" fill="none" stroke="#111111" stroke-width="2.4" stroke-dasharray="4,4" opacity="0.85" pointer-events="none"/>')

        if overlay_audio_duty and f"Audio_Duty_{ch}" in audio_df.columns:
            pts = audio_polyline(audio_df, f"Audio_Duty_{ch}", "duty")
            html.append(f'<polyline points="{pts}" fill="none" stroke="#ff7f0e" stroke-width="2.0" stroke-dasharray="8,4" opacity="0.75" pointer-events="none"/>')

        if overlay_audio_lum and f"Audio_Lum_{ch}" in audio_df.columns:
            pts = audio_polyline(audio_df, f"Audio_Lum_{ch}", "lum")
            html.append(f'<polyline points="{pts}" fill="none" stroke="#555555" stroke-width="2.0" stroke-dasharray="2,5" opacity="0.75" pointer-events="none"/>')

    html.append("</g>")

    handles = points[(points["osc"] == selected_osc) & (points["param"] == selected_param)].sort_values("t")
    for _, row in handles.iterrows():
        if row["t"] < view_start or row["t"] > view_end:
            continue

        x = x_to_px(row["t"])
        y = freq_to_py(row["value"]) if selected_param == "freq" else pct_to_py(row["value"])

        if y < margin_top - 20 or y > height - margin_bottom + 20:
            continue

        html.append(
            f'''
<circle class="drag-point"
        data-point-id="{int(row["point_id"])}"
        cx="{x:.2f}" cy="{y:.2f}" r="7"
        fill="white" stroke="black" stroke-width="2"
        style="cursor:grab;">
  <title>{selected_osc} {selected_param} | t={row["t"]:.1f}, value={row["value"]:.2f}</title>
</circle>
'''
        )

    html.append(
        f'<text x="{margin_left + 8}" y="{margin_top + 15}" font-size="12" fill="#333">'
        f'Selected: {selected_osc} {selected_param} | snap = {float(snap):.1f} s | view {view_start:.1f}-{view_end:.1f} s</text>'
    )

    html.append("</svg>")

    return "\n".join(html)


# ============================================================
# PLOTS
# ============================================================



def plot_preview(points, total_duration, freq_view_mode="sls", playhead_time=None):
    """
    Public-facing preview plot with fixed axis geometry.

    Frequency is on the left axis in true Hz.
    Duty Cycle and Luminance are on the right axis in percent.
    """
    fig, ax_freq = plt.subplots(figsize=(12, 3.9))
    ax_pct = ax_freq.twinx()

    colors = {
        "OSC1": "tab:blue",
        "OSC2": "tab:red",
        "OSC3": "tab:green",
        "OSC4": "tab:purple",
    }

    all_freqs = []

    for osc in OSC_NAMES:
        curve = points[(points["osc"] == osc) & (points["param"] == "freq")].sort_values("t")
        if len(curve) > 0:
            vals = np.asarray(curve["value"].values, dtype=float)
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 0]
            if len(vals) > 0:
                all_freqs.extend(vals.tolist())

    if len(all_freqs) > 0:
        fmin = float(np.min(all_freqs))
        fmax = float(np.max(all_freqs))
        if abs(fmax - fmin) < 0.5:
            pad = max(0.5, fmax * 0.08)
        else:
            pad = max(0.5, (fmax - fmin) * 0.12)
        ylo = max(FREQ_MIN, fmin - pad)
        yhi = min(FREQ_MAX, fmax + pad)
        if yhi <= ylo:
            yhi = ylo + 1
    else:
        ylo, yhi = 5, 60

    for osc in OSC_NAMES:
        curve = points[(points["osc"] == osc) & (points["param"] == "freq")].sort_values("t")
        if len(curve) >= 2:
            ax_freq.plot(
                curve["t"].values,
                curve["value"].values,
                "-",
                color=colors[osc],
                alpha=0.82,
                linewidth=1.55,
                label=f"{osc} Frequency",
            )

        curve = points[(points["osc"] == osc) & (points["param"] == "duty")].sort_values("t")
        if len(curve) >= 2:
            ax_pct.plot(
                curve["t"].values,
                curve["value"].values,
                "--",
                color=colors[osc],
                alpha=0.68,
                linewidth=1.25,
                label=f"{osc} Duty Cycle",
            )

        curve = points[(points["osc"] == osc) & (points["param"] == "lum")].sort_values("t")
        if len(curve) >= 2:
            ax_pct.plot(
                curve["t"].values,
                curve["value"].values,
                ":",
                color=colors[osc],
                alpha=0.78,
                linewidth=1.65,
                label=f"{osc} Luminance",
            )

    sun_curve = points[(points["osc"] == "SUN") & (points["param"] == "lum")].sort_values("t")
    if len(sun_curve) >= 2:
        ax_pct.plot(
            sun_curve["t"].values,
            sun_curve["value"].values,
            "-.",
            color="tab:orange",
            alpha=0.88,
            linewidth=1.75,
            label="SUN Halogen",
        )

    ax_freq.set_xlim(0, total_duration)
    ax_freq.set_ylim(ylo, yhi)
    ax_pct.set_ylim(0, 100)

    ax_freq.set_xlabel("Time (s)")
    ax_freq.set_ylabel("Frequency (Hz)", labelpad=4)
    ax_pct.set_ylabel("Duty Cycle & Luminance (%)", labelpad=4)

    ax_freq.grid(True, alpha=0.22)
    ax_freq.set_title("Sequence Preview")

    lines1, labels1 = ax_freq.get_legend_handles_labels()
    lines2, labels2 = ax_pct.get_legend_handles_labels()
    ax_freq.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        fontsize=7,
        frameon=True,
    )

    # Hard-lock actual plot region to match browser overlay.
    ax_freq.set_position([0.075, 0.30, 0.870, 0.56])
    ax_pct.set_position(ax_freq.get_position())

    return fig


def plot_audio_scrubber(audio_df, freq_view_mode="sls", playhead_time=None):
    """
    Audio-derived strobification plot with fixed axis geometry.

    Browser overlay calibration:
    - data axis left  = 7.5%
    - data axis right = 94.5%
    """
    fig, ax_freq = plt.subplots(figsize=(12, 3.8))

    if audio_df is None or len(audio_df) == 0:
        ax_freq.text(0.5, 0.5, "Upload Audio and press Analyze Audio.", ha="center", va="center")
        ax_freq.set_axis_off()
        ax_freq.set_position([0.075, 0.20, 0.870, 0.66])
        return fig

    fwin = freq_display_window(freq_view_mode)

    freq_colors = {
        1: "tab:blue",
        2: "tab:red",
        3: "tab:green",
        4: "tab:purple",
    }

    for ch in range(1, 5):
        fcol = f"Audio_Freq_{ch}"
        if fcol not in audio_df.columns:
            continue

        y = (freq_to_display(audio_df[fcol].values) - fwin["min_d"]) / (fwin["max_d"] - fwin["min_d"]) * 100

        ax_freq.plot(
            audio_df["Time"],
            y,
            "-",
            linewidth=1.25,
            alpha=0.82,
            color=freq_colors.get(ch, None),
            label=f"Voice {ch} Frequency",
        )

    ax_pct = ax_freq.twinx()

    if "Audio_Amp_1" in audio_df.columns:
        ax_pct.plot(
            audio_df["Time"],
            audio_df["Audio_Amp_1"],
            color="0.30",
            linestyle="-.",
            linewidth=1.05,
            alpha=0.72,
            label="Voice 1 Amplitude",
        )

    if "Audio_Duty_1" in audio_df.columns:
        ax_pct.plot(
            audio_df["Time"],
            audio_df["Audio_Duty_1"],
            color="tab:orange",
            linestyle="--",
            linewidth=1.15,
            alpha=0.78,
            label="Voice 1 Duty Cycle",
        )

    if "Audio_Lum_1" in audio_df.columns:
        ax_pct.plot(
            audio_df["Time"],
            audio_df["Audio_Lum_1"],
            color="tab:gray",
            linestyle=":",
            linewidth=1.65,
            alpha=0.84,
            label="Voice 1 Luminance",
        )

    ax_freq.set_ylim(0, 100)
    ax_pct.set_ylim(0, 100)

    ax_freq.set_xlabel("Time (s)")
    ax_freq.set_ylabel(f"Frequency Display\n{fwin['label']}", labelpad=4)
    ax_pct.set_ylabel("Amplitude, Duty Cycle & Luminance (%)", labelpad=4)

    ax_freq.grid(True, alpha=0.22)
    ax_freq.set_title("Audio Scrubber Plot")

    lines1, labels1 = ax_freq.get_legend_handles_labels()
    lines2, labels2 = ax_pct.get_legend_handles_labels()
    ax_freq.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        fontsize=7,
        frameon=True,
    )

    # Hard-lock actual plot region to match browser overlay.
    ax_freq.set_position([0.075, 0.30, 0.870, 0.56])
    ax_pct.set_position(ax_freq.get_position())

    return fig


# ============================================================
# UI
# ============================================================

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(
            """
            body { background: #202327; }
            .container-fluid { padding-left: 12px; padding-right: 12px; }
            .main-card {
              background: #ffffff;
              border-radius: 12px;
              padding: 14px;
              box-shadow: 0 2px 10px rgba(0,0,0,0.22);
              margin-bottom: 14px;
              border: 1px solid rgba(0,0,0,0.06);
            }
            .small-note { font-size: 12px; color: #666; }
            .coord-box {
              font-family: monospace;
              background: #f2f2f2;
              padding: 7px;
              border-radius: 6px;
              min-height: 34px;
              font-size: 12px;
            }
            .btn-wide { width: 100%; margin-bottom: 5px; }
            #client_coord {
              position: fixed;
              right: 20px;
              bottom: 18px;
              background: rgba(30, 30, 30, 0.88);
              color: white;
              font-family: monospace;
              font-size: 13px;
              padding: 7px 10px;
              border-radius: 6px;
              z-index: 9999;
              pointer-events: none;
            }
            .form-group { margin-bottom: 10px; }
            h4 { margin-top: 4px; margin-bottom: 8px; }

            #calculation_guide_card { display: none !important; }

            .readme-card {
              background: #ffffff;
              border-radius: 10px;
              padding: 16px 18px;
              box-shadow: 0 1px 5px rgba(0,0,0,0.12);
              margin-bottom: 14px;
              max-width: 1180px;
            }

            .readme-card h4 {
              margin-top: 0;
              margin-bottom: 10px;
              color: #1f3f5f;
            }

            .readme-card h5 {
              margin-top: 16px;
              margin-bottom: 6px;
              color: #303030;
              font-size: 15px;
            }

            .readme-card p {
              font-size: 13px;
              line-height: 1.45;
              color: #444;
              margin-bottom: 8px;
            }

            .readme-card ul {
              padding-left: 20px;
              margin-top: 6px;
              margin-bottom: 10px;
            }

            .readme-card li {
              font-size: 13px;
              line-height: 1.42;
              color: #444;
              margin-bottom: 4px;
            }

            .summary-grid {
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
              gap: 10px;
              margin-top: 10px;
              margin-bottom: 10px;
            }

            .summary-tile {
              border: 1px solid #dce6ef;
              background: #f8fbfe;
              border-radius: 8px;
              padding: 10px 12px;
            }

            .summary-tile-title {
              font-weight: 700;
              font-size: 12px;
              color: #24496f;
              text-transform: uppercase;
              letter-spacing: 0.03em;
              margin-bottom: 4px;
            }

            .summary-tile-value {
              font-size: 13px;
              color: #222;
              line-height: 1.35;
            }

            .readme-pill {
              display: inline-block;
              font-family: monospace;
              font-size: 11px;
              background: #eef4fb;
              color: #24496f;
              border: 1px solid #d5e4f4;
              border-radius: 999px;
              padding: 2px 7px;
              margin: 2px 3px 2px 0;
            }

            .readme-warning {
              border-left: 4px solid #e09f2f;
              background: #fff8ed;
              padding: 9px 11px;
              border-radius: 6px;
              margin-top: 10px;
              font-size: 13px;
              color: #4a3a20;
            }


            .info-card {
              border-left: 4px solid #3b6ea8;
              background: #ffffff;
            }

            .info-card h4 {
              color: #24496f;
              margin-bottom: 8px;
            }

            .info-card h5 {
              font-size: 13px;
              margin-top: 10px;
              margin-bottom: 4px;
              color: #333;
            }

            .info-card p,
            .info-card li {
              font-size: 12px;
              line-height: 1.35;
              color: #555;
            }

            .info-card ul {
              padding-left: 18px;
              margin-top: 4px;
              margin-bottom: 8px;
            }

            .info-pill {
              display: inline-block;
              font-family: monospace;
              font-size: 11px;
              background: #eef4fb;
              color: #24496f;
              border: 1px solid #d5e4f4;
              border-radius: 999px;
              padding: 2px 7px;
              margin: 2px 2px 2px 0;
            }

            .playhead-wrap {
              position: relative;
              width: 100%;
            }

            .browser-playhead {
              position: absolute;
              top: 0;
              bottom: 0;
              left: var(--playhead-left, 0%);
              width: 2px;
              background: rgba(220, 0, 0, 0.85);
              z-index: 20;
              pointer-events: none;
              display: none;
              box-shadow: 0 0 4px rgba(220, 0, 0, 0.35);
            }

            .browser-playhead-label {
              position: absolute;
              top: 4px;
              transform: translateX(5px);
              background: rgba(220, 0, 0, 0.85);
              color: white;
              font-family: monospace;
              font-size: 11px;
              padding: 2px 4px;
              border-radius: 3px;
              white-space: nowrap;
            }
            """
        ),
        ui.tags.script(
            """
            function roundToGrid(x, grid) {
              return Math.round(x / grid) * grid;
            }

            function getSnap(svg) {
              if (!svg) return 0.1;
              const snap = parseFloat(svg.dataset.snap);
              if (isNaN(snap)) return 0.1;
              return Math.max(0.1, snap);
            }

            function cleanValue(param, value) {
              if (param === 'freq') {
                value = Math.max(0.01, Math.min(200.00, value));
                return Math.round(value * 100) / 100;
              }
              if (param === 'duty') {
                value = Math.max(1, Math.min(99, value));
                return Math.round(value);
              }
              if (param === 'lum') {
                value = Math.max(0, Math.min(100, value));
                return Math.round(value);
              }
              return value;
            }

            function freqToDisplay(freq) {
              freq = Math.max(0.01, Math.min(200.00, freq));
              if (freq <= 5) {
                return (freq - 0.01) / (5 - 0.01) * 10;
              } else if (freq <= 8) {
                return 10 + (freq - 5) / (8 - 5) * 12;
              } else if (freq <= 15) {
                return 22 + (freq - 8) / (15 - 8) * 48;
              } else if (freq <= 20) {
                return 70 + (freq - 15) / (20 - 15) * 12;
              } else if (freq <= 30) {
                return 82 + (freq - 20) / (30 - 20) * 8;
              } else if (freq <= 60) {
                return 90 + (freq - 30) / (60 - 30) * 7;
              } else {
                return 97 + (freq - 60) / (200 - 60) * 3;
              }
            }

            function displayToFreq(displayValue) {
              displayValue = Math.max(0, Math.min(100, displayValue));
              let freq;
              if (displayValue <= 10) {
                freq = 0.01 + displayValue / 10 * (5 - 0.01);
              } else if (displayValue <= 22) {
                freq = 5 + (displayValue - 10) / 12 * (8 - 5);
              } else if (displayValue <= 70) {
                freq = 8 + (displayValue - 22) / 48 * (15 - 8);
              } else if (displayValue <= 82) {
                freq = 15 + (displayValue - 70) / 12 * (20 - 15);
              } else if (displayValue <= 90) {
                freq = 20 + (displayValue - 82) / 8 * (30 - 20);
              } else if (displayValue <= 97) {
                freq = 30 + (displayValue - 90) / 7 * (60 - 30);
              } else {
                freq = 60 + (displayValue - 97) / 3 * (200 - 60);
              }
              freq = Math.max(0.01, Math.min(200.00, freq));
              return Math.round(freq * 100) / 100;
            }

            function svgToData(svg, clientX, clientY) {
              const rect = svg.getBoundingClientRect();
              const width = svg.viewBox.baseVal.width;
              const height = svg.viewBox.baseVal.height;

              const pxRaw = (clientX - rect.left) * (width / rect.width);
              const pyRaw = (clientY - rect.top) * (height / rect.height);

              const left = parseFloat(svg.dataset.plotLeft);
              const right = parseFloat(svg.dataset.plotRight);
              const top = parseFloat(svg.dataset.plotTop);
              const bottom = parseFloat(svg.dataset.plotBottom);

              const totalDuration = parseFloat(svg.dataset.totalDuration);
              const viewStart = parseFloat(svg.dataset.viewStart);
              const viewEnd = parseFloat(svg.dataset.viewEnd);
              const viewWidth = viewEnd - viewStart;
              const selectedParam = svg.dataset.selectedParam;
              const snap = getSnap(svg);

              const freqDisplayMin = parseFloat(svg.dataset.freqDisplayMin);
              const freqDisplayMax = parseFloat(svg.dataset.freqDisplayMax);
              const pctMin = parseFloat(svg.dataset.pctMin);
              const pctMax = parseFloat(svg.dataset.pctMax);

              let t = viewStart + (pxRaw - left) / (right - left) * viewWidth;
              t = Math.max(0, Math.min(totalDuration, t));
              t = roundToGrid(t, snap);
              t = Math.max(0, Math.min(totalDuration, t));
              t = Math.round(t * 10) / 10;

              let value;
              if (selectedParam === 'freq') {
                let displayValue = freqDisplayMin +
                  (1 - ((pyRaw - top) / (bottom - top))) *
                  (freqDisplayMax - freqDisplayMin);
                value = displayToFreq(displayValue);
              } else {
                value = pctMax - ((pyRaw - top) / (bottom - top)) * (pctMax - pctMin);
              }

              value = cleanValue(selectedParam, value);

              let px = left + ((t - viewStart) / viewWidth) * (right - left);
              let py;

              if (selectedParam === 'freq') {
                let displayValue = freqToDisplay(value);
                py = top + (1 - ((displayValue - freqDisplayMin) / (freqDisplayMax - freqDisplayMin))) *
                  (bottom - top);
              } else {
                py = top + (1 - ((value - pctMin) / (pctMax - pctMin))) * (bottom - top);
              }

              return { t: t, value: value, px: px, py: py };
            }

            function updateClientCoord(coords) {
              const box = document.getElementById('client_coord');
              if (!box) return;
              box.textContent = 'x = ' + coords.t.toFixed(1) + ' s | y = ' + coords.value;
            }

            let draggingPoint = null;
            let draggingElement = null;
            let lastDragCoords = null;
            let lastMouseSend = 0;

            document.addEventListener('mousedown', function(e) {
              const target = e.target;
              if (target.classList && target.classList.contains('drag-point')) {
                draggingPoint = target.dataset.pointId;
                draggingElement = target;

                const svg = document.getElementById('curve_svg');
                const coords = svgToData(svg, e.clientX, e.clientY);

                lastDragCoords = coords;
                updateClientCoord(coords);

                draggingElement.setAttribute('cx', coords.px);
                draggingElement.setAttribute('cy', coords.py);
                draggingElement.setAttribute('fill', '#ffd34d');

                Shiny.setInputValue('point_select', {
                  point_id: parseInt(draggingPoint),
                  t: coords.t,
                  value: coords.value,
                  nonce: Math.random()
                }, {priority: 'event'});

                e.preventDefault();
              }
            });

            document.addEventListener('mousemove', function(e) {
              const svg = document.getElementById('curve_svg');
              if (!svg) return;

              const rect = svg.getBoundingClientRect();

              if (
                e.clientX >= rect.left &&
                e.clientX <= rect.right &&
                e.clientY >= rect.top &&
                e.clientY <= rect.bottom
              ) {
                const coords = svgToData(svg, e.clientX, e.clientY);
                updateClientCoord(coords);

                const now = Date.now();
                if (now - lastMouseSend > 120) {
                  lastMouseSend = now;
                  Shiny.setInputValue('mouse_pos', {
                    t: coords.t,
                    value: coords.value,
                    nonce: Math.random()
                  }, {priority: 'event'});
                }

                if (draggingPoint !== null && draggingElement !== null) {
                  lastDragCoords = coords;
                  draggingElement.setAttribute('cx', coords.px);
                  draggingElement.setAttribute('cy', coords.py);
                }
              }
            });

            document.addEventListener('mouseup', function(e) {
              if (draggingPoint !== null && lastDragCoords !== null) {
                Shiny.setInputValue('point_drag_final', {
                  point_id: parseInt(draggingPoint),
                  t: lastDragCoords.t,
                  value: lastDragCoords.value,
                  nonce: Math.random()
                }, {priority: 'event'});
              }
              draggingPoint = null;
              draggingElement = null;
              lastDragCoords = null;
            });

            document.addEventListener('click', function(e) {
              const svg = document.getElementById('curve_svg');
              if (!svg) return;

              if (e.target.classList && e.target.classList.contains('drag-point')) {
                return;
              }

              if (e.target.id === 'curve_svg' || e.target.closest('#curve_svg')) {
                const coords = svgToData(svg, e.clientX, e.clientY);
                updateClientCoord(coords);

                Shiny.setInputValue('svg_click', {
                  t: coords.t,
                  value: coords.value,
                  nonce: Math.random()
                }, {priority: 'event'});
              }
            });

            function attachBrowserSidePlayhead() {
              const audio = document.getElementById('uploaded_audio_player');
              if (!audio) return;
              if (audio.dataset.browserPlayheadBound === '1') return;

              audio.dataset.browserPlayheadBound = '1';

              let rafId = null;

              function formatTime(t) {
                if (!isFinite(t)) return "0.0 s";
                return t.toFixed(1) + " s";
              }

              function pctBetweenPlotBounds(wrap, pct) {
                const styles = getComputedStyle(wrap);

                let left = styles.getPropertyValue('--plot-left').trim();
                let right = styles.getPropertyValue('--plot-right').trim();

                let leftNum = parseFloat(left);
                let rightNum = parseFloat(right);

                if (!isFinite(leftNum)) leftNum = 0;
                if (!isFinite(rightNum)) rightNum = 100;

                pct = Math.max(0, Math.min(100, pct));

                return leftNum + (pct / 100.0) * (rightNum - leftNum);
              }

              function ensurePlayheadElement(wrap) {
                let line = wrap.querySelector('.browser-playhead');
                if (!line) {
                  line = document.createElement('div');
                  line.className = 'browser-playhead';

                  const label = document.createElement('div');
                  label.className = 'browser-playhead-label';
                  label.textContent = '0.0 s';
                  line.appendChild(label);

                  wrap.appendChild(line);
                }
                return line;
              }

              function updateStaticPlotPlayheads() {
                const duration = audio.duration || 0;
                const t = audio.currentTime || 0;
                const wraps = document.querySelectorAll('.playhead-wrap');

                if (!duration || !isFinite(duration) || wraps.length === 0) {
                  wraps.forEach(function(wrap) {
                    const line = ensurePlayheadElement(wrap);
                    line.style.display = 'none';
                  });
                  return;
                }

                const rawPct = Math.max(0, Math.min(100, (t / duration) * 100));

                wraps.forEach(function(wrap) {
                  const line = ensurePlayheadElement(wrap);
                  const label = line.querySelector('.browser-playhead-label');
                  const calibratedPct = pctBetweenPlotBounds(wrap, rawPct);

                  line.style.display = 'block';
                  line.style.left = calibratedPct + '%';

                  if (label) {
                    label.textContent = formatTime(t);
                  }
                });
              }

              function updateSvgEditorPlayhead() {
                const svg = document.getElementById('curve_svg');
                if (!svg) return;

                const duration = audio.duration || 0;
                const t = audio.currentTime || 0;

                let line = svg.querySelector('#browser_svg_playhead');
                let label = svg.querySelector('#browser_svg_playhead_label');

                if (!line) {
                  line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                  line.setAttribute('id', 'browser_svg_playhead');
                  line.setAttribute('stroke', 'rgba(220,0,0,0.9)');
                  line.setAttribute('stroke-width', '2.4');
                  line.setAttribute('pointer-events', 'none');
                  line.setAttribute('opacity', '0.95');
                  svg.appendChild(line);
                }

                if (!label) {
                  label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                  label.setAttribute('id', 'browser_svg_playhead_label');
                  label.setAttribute('font-size', '11');
                  label.setAttribute('font-family', 'monospace');
                  label.setAttribute('fill', 'rgba(220,0,0,0.95)');
                  label.setAttribute('pointer-events', 'none');
                  svg.appendChild(label);
                }

                if (!duration || !isFinite(duration)) {
                  line.setAttribute('display', 'none');
                  label.setAttribute('display', 'none');
                  return;
                }

                const viewStart = parseFloat(svg.dataset.viewStart);
                const viewEnd = parseFloat(svg.dataset.viewEnd);
                const plotLeft = parseFloat(svg.dataset.plotLeft);
                const plotRight = parseFloat(svg.dataset.plotRight);
                const plotTop = parseFloat(svg.dataset.plotTop);
                const plotBottom = parseFloat(svg.dataset.plotBottom);

                if (!isFinite(viewStart) || !isFinite(viewEnd) || viewEnd <= viewStart) {
                  line.setAttribute('display', 'none');
                  label.setAttribute('display', 'none');
                  return;
                }

                if (t < viewStart || t > viewEnd) {
                  line.setAttribute('display', 'none');
                  label.setAttribute('display', 'none');
                  return;
                }

                const x = plotLeft + ((t - viewStart) / (viewEnd - viewStart)) * (plotRight - plotLeft);

                line.setAttribute('display', 'block');
                line.setAttribute('x1', x);
                line.setAttribute('x2', x);
                line.setAttribute('y1', plotTop);
                line.setAttribute('y2', plotBottom);

                label.setAttribute('display', 'block');
                label.setAttribute('x', Math.min(x + 5, plotRight - 45));
                label.setAttribute('y', plotTop + 14);
                label.textContent = formatTime(t);
              }

              function updateAllPlayheads() {
                updateStaticPlotPlayheads();
                updateSvgEditorPlayhead();
              }

              function loop() {
                updateAllPlayheads();

                if (!audio.paused && !audio.ended) {
                  rafId = window.requestAnimationFrame(loop);
                } else {
                  rafId = null;
                }
              }

              function startLoop() {
                if (rafId === null) {
                  rafId = window.requestAnimationFrame(loop);
                }
              }

              function stopLoopButUpdate() {
                if (rafId !== null) {
                  window.cancelAnimationFrame(rafId);
                  rafId = null;
                }
                updateAllPlayheads();
              }

              audio.addEventListener('play', function() {
                startLoop();
              });

              audio.addEventListener('pause', function() {
                stopLoopButUpdate();
              });

              audio.addEventListener('ended', function() {
                stopLoopButUpdate();
              });

              audio.addEventListener('seeked', function() {
                stopLoopButUpdate();
              });

              audio.addEventListener('loadedmetadata', function() {
                updateAllPlayheads();
              });

              audio.addEventListener('loadeddata', function() {
                updateAllPlayheads();
              });

              // Keep the SVG editor playhead aligned after Shiny redraws the SVG
              // because of zoom, pan, curve edits, or input changes.
              setInterval(function() {
                updateAllPlayheads();
              }, 500);

              updateAllPlayheads();
            }

            setInterval(attachBrowserSidePlayhead, 500);


            function tidyUploadCompleteText() {
              const nodes = document.querySelectorAll('body *');
              nodes.forEach(function(el) {
                if (!el || !el.textContent) return;
                const txt = el.textContent.trim();
                if (txt === 'Upload complete' || txt === 'upload complete' || txt === 'UPLOAD COMPLETE') {
                  el.textContent = 'UPLOAD COMPLETE!';
                }
              });
            }

            setInterval(tidyUploadCompleteText, 500);

	
            """
        ),
    ),
    ui.tags.div({"id": "client_coord"}, "x = -- | y = --"),
    ui.h2("RoXiva RX1 Curve Editor + Audio Scrubber"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                {"class": "main-card"},
                ui.input_text("sequence_name", "Sequence Name", "my_drawn_roxiva_sequence"),
                ui.input_numeric("total_duration", "Total Duration (s)", value=60, min=1, step=1),
                ui.input_select(
                    "time_snap_mode",
                    "Time Snap / Export Rule",
                    {
                        "auto": "Duration-Based Snap",
                        "1.0": "1.0 s Snap",
                        "0.5": "0.5 s Snap",
                        "0.3": "0.3 s Snap",
                        "0.2": "0.2 s Snap",
                        "0.1": "0.1 s Snap",
                    },
                    selected="auto",
                ),
                ui.div({"class": "small-note"}, ui.output_text("snap_text")),
                ui.input_select(
                    "export_motion_mode",
                    "Export Motion Mode",
                    {
                        "hold": "Hold Each Step",
                        "interpolate": "Interpolate Between Points (STP > 0.1 s only)",
                    },
                    selected="hold",
                ),
                ui.div({"class": "small-note"}, ""),
                ui.hr(),
                ui.input_select(
                    "freq_view_mode",
                    "Frequency Display",
                    {
                        "sls": "SLS Design: 5-60 Hz",
                        "alpha": "Alpha Focus: 5-20 Hz",
                        "low": "Low Range: 0.01-30 Hz",
                        "full": "Full RX1: 0.01-200 Hz",
                    },
                    selected="sls",
                ),
                ui.hr(),
                ui.h4("Time Zoom"),
                ui.row(
                    ui.column(6, ui.input_action_button("zoom_in", "Zoom In", class_="btn-wide")),
                    ui.column(6, ui.input_action_button("zoom_out", "Zoom Out", class_="btn-wide")),
                ),
                ui.row(
                    ui.column(6, ui.input_action_button("pan_left", "← Left", class_="btn-wide")),
                    ui.column(6, ui.input_action_button("pan_right", "Right →", class_="btn-wide")),
                ),
                ui.input_action_button("zoom_reset", "Show Full Sequence", class_="btn-wide"),
                ui.div({"class": "small-note"}, ui.output_text("zoom_text")),
                ui.hr(),
                ui.input_select("selected_osc", "Oscillator", EDITOR_OSC_NAMES, selected="OSC1"),
                ui.input_select(
                    "selected_param",
                    "Parameter",
                    {"freq": "Frequency", "duty": "Duty", "lum": "Luminance"},
                    selected="freq",
                ),
            ),
            ui.div(
                {"class": "main-card"},
                ui.input_action_button("add_point_mode", "Add Point by Clicking Plot", class_="btn-wide"),
                ui.input_action_button("delete_point", "Delete Selected Point", class_="btn-wide"),
                ui.input_action_button("copy_curve_all", "Copy Selected Curve to All OSCs", class_="btn-wide"),
                ui.input_action_button("reset_curve", "Reset Selected Curve", class_="btn-wide"),
            ),
            ui.div(
                {"class": "main-card"},
                ui.h4("Selected Point"),
                ui.input_numeric("selected_t", "Time (s)", value=0, min=0, step=0.1),
                ui.input_numeric("selected_value", "Value", value=0, step=0.1),
                ui.input_action_button("apply_selected_values", "Apply Typed Values", class_="btn-wide"),
                ui.div({"class": "small-note"}, "Values are clamped to RX1 limits."),
            ),
            ui.div(
                {"class": "main-card"},
                ui.h4("Audio Import / Scrubber"),
                ui.input_file("audio_file", "Upload Audio File", accept=[".wav", ".mp3", ".flac", ".ogg"]),
                ui.input_select(
                    "audio_engine",
                    "Extraction Engine",
                    {
                        "fft": "FFT Peaks",
                        "harmonic_fft": "FFT Harmonic Salience",
                        "cqt": "CQT Peaks",
                    },
                    selected="fft",
                ),
                ui.input_numeric("audio_channels", "Voices to Extract", value=1, min=1, max=4, step=1),
                ui.input_numeric("audio_step", "Audio Step Duration (s)", value=0.2, min=0.1, max=5, step=0.1),
                ui.input_numeric("audio_n_fft", "FFT Size", value=4096, min=512, max=16384, step=512),
                ui.input_numeric("audio_band_lo", "Band Low (Hz)", value=40, min=1, max=2000, step=1),
                ui.input_numeric("audio_band_hi", "Band High (Hz)", value=6000, min=100, max=20000, step=100),
                ui.input_numeric("audio_peak_rel", "Peak Threshold", value=0.20, min=0.01, max=1.0, step=0.01),
                ui.input_select(
                    "audio_mapping",
                    "Frequency Mapping",
                    {
                        "none": "None",
                        "rx1_safe": "Fold to RX1",
                        "sls_fold": "Fold to SLS",
                        "sls_center": "Fold Near 11.56 Hz",
                        "alpha_array": "Fold to Alpha",
                    },
                    selected="sls_center",
                ),
                ui.input_checkbox("audio_snap_music", "Snap to Musical Grid", value=False),
                ui.input_select("audio_interval", "Interval Transposition", list(INTERVALS.keys()), selected="None / Unison"),
                ui.input_select("audio_interval_dir", "Interval Direction", {"up": "Up", "down": "Down"}, selected="up"),
                ui.hr(),
                ui.input_select(
                    "audio_amp_norm",
                    "Amplitude Normalization",
                    {
                        "global_linear": "Global Linear",
                        "per_voice_linear": "Per-Voice Linear",
                        "percentile_clip": "Percentile Clipped",
                        "log_global": "Log-Compressed Global",
                        "sqrt_global": "SqRt-Compressed Global",
                    },
                    selected="global_linear",
                ),
                ui.input_numeric("audio_amp_percentile", "Percentile Clip", value=95, min=50, max=100, step=1),
                ui.hr(),
                ui.input_select(
                    "audio_duty_method",
                    "Duty Cycle Calculation",
                    {
                        "fixed": "Fixed Duty",
                        "amplitude": "Amplitude Mapped",
                        "inverted_amplitude": "Inverted Amplitude",
                        "wide_at_high_amp": "Wider Duty : Higher Amplitude",
                        "narrow_at_high_amp": "Narrower Duty : Higher Amplitude",
                        "occupancy_proxy": "Spectral Occupancy Proxy",
                        "harmonic_band_proxy": "Harmonic-Band Proxy",
                        "amp_gated_occupancy": "Amplitude-Gated Occupancy",
                    },
                    selected="fixed",
                ),
                ui.input_numeric("audio_fixed_duty", "Fixed Duty (%)", value=50, min=1, max=99, step=1),
                ui.input_numeric("audio_duty_floor", "Duty Floor (%)", value=10, min=1, max=99, step=1),
                ui.input_numeric("audio_duty_ceiling", "Duty Ceiling (%)", value=90, min=1, max=99, step=1),
                ui.input_numeric("audio_duty_amp_gate", "Duty Amp Gate (%)", value=0, min=0, max=100, step=1),
                ui.hr(),
                ui.input_select(
                    "audio_lum_method",
                    "Luminance Calculation",
                    {
                        "amplitude": "Amplitude Mapped",
                        "sqrt_amplitude": "SqRt-Shaped Amplitude",
                        "log_amplitude": "Log-Shaped Amplitude",
                        "threshold_gated": "Threshold-Gated Amplitude",
                        "fixed": "Fixed Luminance",
                    },
                    selected="amplitude",
                ),
                ui.input_numeric("audio_fixed_lum", "Fixed Luminance (%)", value=50, min=0, max=100, step=1),
                ui.input_numeric("audio_lum_floor", "Luminance Floor (%)", value=0, min=0, max=100, step=1),
                ui.input_numeric("audio_lum_ceiling", "Luminance Ceiling (%)", value=100, min=0, max=100, step=1),
                ui.input_numeric("audio_lum_amp_gate", "Luminance Amp Gate (%)", value=0, min=0, max=100, step=1),
                ui.hr(),
                ui.input_numeric("audio_occ_bw", "Occupancy Bandwidth (Cents)", value=35, min=5, max=300, step=5),
                ui.input_select(
                    "audio_smoothing",
                    "Output Smoothing",
                    {
                        "none": "None",
                        "rolling_mean": "Rolling Mean",
                        "rolling_median": "Rolling Median",
                    },
                    selected="none",
                ),
                ui.input_numeric("audio_smooth_window", "Smoothing Window (Frames)", value=1, min=1, max=51, step=2),
                ui.input_action_button("analyze_audio", "Analyze Audio", class_="btn-wide"),
                ui.div({"class": "small-note"}, ui.output_text("audio_status")),
            ),
            ui.div(
                {"class": "main-card"},
                ui.h4("Audio Overlay / Apply"),
                ui.input_numeric("overlay_voice", "Overlay / Apply Voice", value=1, min=1, max=4, step=1),
                ui.input_checkbox("overlay_audio_freq", "Overlay Audio Frequency", value=True),
                ui.input_checkbox("overlay_audio_duty", "Overlay Audio Duty Cycle", value=False),
                ui.input_checkbox("overlay_audio_lum", "Overlay Audio Luminance", value=False),
                ui.input_action_button("apply_audio_selected_freq", "Apply Audio Frequency to Selected OSC", class_="btn-wide"),
                ui.input_action_button("apply_audio_selected_all", "Apply Audio Frequency + Duty + Luminance to Selected OSC", class_="btn-wide"),
                ui.input_action_button("apply_audio_all_oscs", "Apply Audio Voices 1?4 to OSC1?4", class_="btn-wide"),
            ),
            ui.div(
                {"class": "main-card"},
                ui.h4("Save / Load"),
                ui.download_button("download_txt", "Download RX1 TXT", class_="btn-wide"),
                ui.download_button("download_project", "Download Editable JSON", class_="btn-wide"),
                ui.download_button("download_png", "Download Preview PNG", class_="btn-wide"),
                ui.download_button("download_audio_csv", "Download Audio CSV", class_="btn-wide"),
                ui.hr(),
                ui.h4("LUCiO / Lucia Export"),
                ui.input_file("lucio_template", "Upload LUCiO Template .lscf", accept=[".lscf"]),
                ui.input_numeric("lucio_control_step", "LUCiO Control Step (s)", value=1.0, min=0.1, max=10, step=0.1),
                ui.download_button("download_lucio_lscf", "Download LUCiO .LSCF", class_="btn-wide"),
                ui.download_button("download_lucio_debug", "Download LUCiO Debug CSV", class_="btn-wide"),
                ui.div({"class": "small-note"}, "For SUN/halogen, select OSC = SUN and Parameter = Luminance, then draw a 0-100 curve."),
                ui.hr(),
                ui.input_file("upload_project", "Reload Editable JSON", accept=[".json"]),
            ),

            ui.div(
                {"class": "main-card info-card", "id": "calculation_guide_card"},
                ui.h4("Calculation Guide"),
                ui.p("These notes summarise how the current audio-to-strobe controls shape the generated RX1 sequence."),
                ui.h5("Extraction Engine"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.strong("FFT Peaks: "), "fast spectral peak extraction; best first test for most WAV files."),
                    ui.tags.li(ui.tags.strong("Harmonic FFT: "), "ranks candidates using energy at harmonic partials, favouring more tonally stable sources."),
                    ui.tags.li(ui.tags.strong("CQT Peaks: "), "music-friendly semitone-spaced analysis; slower, but useful for pitched material."),
                ),
                ui.h5("Frequency Mapping"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.strong("SLS Fold / Centre: "), "folds audio-derived frequencies by octaves into the stroboscopic range."),
                    ui.tags.li(ui.tags.strong("Alpha Array: "), "compresses candidates toward the alpha-like note grid."),
                    ui.tags.li(ui.tags.strong("Interval Transposition: "), "shifts extracted candidates by musical intervals before final range mapping."),
                ),
                ui.h5("Amplitude, Duty Cycle & Luminance"),
                ui.tags.ul(
                    ui.tags.li(ui.tags.strong("Amplitude Normalisation: "), "sets how raw spectral strength becomes a 0?100 control signal."),
                    ui.tags.li(ui.tags.strong("Duty Cycle: "), "controls the on/off proportion of each flicker cycle."),
                    ui.tags.li(ui.tags.strong("Luminance: "), "controls RX1 brightness/intensity after amplitude shaping or gating."),
                ),
                ui.h5("Line Styles"),
                ui.div(
                    ui.tags.span({"class": "info-pill"}, "Frequency = Solid"),
                    ui.tags.span({"class": "info-pill"}, "Duty Cycle = Dashed"),
                    ui.tags.span({"class": "info-pill"}, "Luminance = Dotted"),
                    ui.tags.span({"class": "info-pill"}, "Amplitude = Dash-Dot"),
                ),
                ui.p("For public demonstrations, start with FFT Peaks, 1?2 voices, 0.2?0.5 s step duration, and short WAV files."),
            ),

            width=330,
        ),
        ui.div(
            {"class": "main-card"},
            ui.row(
                ui.column(4, ui.strong("Live Cursor")),
                ui.column(4, ui.strong("Export Estimate")),
                ui.column(4, ui.strong("Status")),
            ),
            ui.row(
                ui.column(4, ui.div({"class": "coord-box"}, ui.output_text("mouse_text"))),
                ui.column(4, ui.div({"class": "coord-box"}, ui.output_text("line_count_text"))),
                ui.column(4, ui.div({"class": "coord-box"}, ui.output_text("status_text"))),
            ),
        ),
                ui.div({"class": "main-card"}, ui.output_ui("svg_editor")),

        ui.div(
            {"class": "main-card"},
            ui.h4("Audio Playback"),
            ui.output_ui("audio_player_ui"),
            ui.div(
                {"class": "small-note"},
                "Use the player to hear the uploaded file. The red vertical cursor tracks playback time."
            ),
        ),

        ui.div(
            {"class": "main-card"},
            ui.h4("Audio Scrubber Plot"),
            ui.div(
                {"class": "playhead-wrap", "data-playhead-target": "audio_plot", "style": "--plot-left: 12.0%; --plot-right: 94.5%;"},
                ui.output_plot("audio_plot", height="340px"),
                ui.tags.div(
                    {"class": "browser-playhead"},
                    ui.tags.div({"class": "browser-playhead-label"}, "0.0 s")
                ),
            )
        ),

        ui.div(
            {"class": "main-card"},
            ui.h4("Waveform / Spectrogram View"),
            ui.div(
                {"class": "playhead-wrap", "data-playhead-target": "audio_waveform_spectrogram_plot", "style": "--plot-left: 7.5%; --plot-right: 94.5%;"},
                ui.output_plot("audio_waveform_spectrogram_plot", height="520px"),
                ui.tags.div(
                    {"class": "browser-playhead"},
                    ui.tags.div({"class": "browser-playhead-label"}, "0.0 s")
                ),
            )
        ),

        ui.div(
            {"class": "main-card"},
            ui.h4("Sequence Preview Plot"),
            ui.div(
                {"class": "playhead-wrap", "data-playhead-target": "preview_plot", "style": "--plot-left: 7.3%; --plot-right: 94.5%;"},
                ui.output_plot("preview_plot", height="360px"),
                ui.tags.div(
                    {"class": "browser-playhead"},
                    ui.tags.div({"class": "browser-playhead-label"}, "0.0 s")
                ),
            )
        ),


        ui.div(
            {"class": "readme-card"},
            ui.h4("Current Configuration Summary"),
            ui.output_ui("current_configuration_ui"),
        ),

        ui.div(
            {"class": "readme-card"},
            ui.h4("Control Reference / README"),
            ui.output_ui("control_reference_ui"),
        ),

    ),
)


# ============================================================
# SERVER
# ============================================================


# ============================================================
# PUBLIC README / OPTION EXPLANATION HELPERS
# ============================================================

AUDIO_ENGINE_LABELS = {
    "fft": "FFT Peaks",
    "harmonic_fft": "FFT Harmonic Salience",
    "cqt": "CQT Peaks",
}

AUDIO_MAPPING_LABELS = {
    "none": "None",
    "rx1_safe": "Fold to RX1",
    "sls_fold": "Fold to SLS",
    "sls_center": "Fold Near 11.56 Hz",
    "alpha_array": "Fold to Alpha",
}

AMP_NORM_LABELS = {
    "global_linear": "Global Linear",
    "per_voice_linear": "Per-Voice Linear",
    "percentile_clip": "Percentile Clipped",
    "log_global": "Log-Compressed Global",
    "sqrt_global": "SqRt-Compressed Global",
}

DUTY_LABELS = {
    "fixed": "Fixed Duty",
    "amplitude": "Amplitude Mapped",
    "inverted_amplitude": "Inverted Amplitude",
    "wide_at_high_amp": "Wider Duty : Higher Amplitude",
    "narrow_at_high_amp": "Narrower Duty : Higher Amplitude",
    "occupancy_proxy": "Spectral Occupancy Proxy",
    "harmonic_band_proxy": "Harmonic-Band Proxy",
    "amp_gated_occupancy": "Amplitude-Gated Occupancy",
}

LUM_LABELS = {
    "amplitude": "Amplitude Mapped",
    "sqrt_amplitude": "SqRt-Shaped Amplitude",
    "log_amplitude": "Log-Shaped Amplitude",
    "threshold_gated": "Threshold-Gated Amplitude",
    "fixed": "Fixed Luminance",
}

SMOOTH_LABELS = {
    "none": "None",
    "rolling_mean": "Rolling Mean",
    "rolling_median": "Rolling Median",
}

def option_label(value, labels):
    return labels.get(value, str(value))




def explain_extraction_engine(engine):
    return {
        "fft": "FFT Peaks extracts the strongest spectral peaks in each analysis frame. It is fast, transparent, and useful for diagnostics or simple audio-following, but it does not include the harmonic-support ranking used by Chiptune 3.DX.",
        "harmonic_fft": "FFT Harmonic Salience is the Chiptune 3.DX-style extraction mode. It first identifies FFT peak candidates, then ranks them by energy at harmonic partials. This favours candidates with stronger tonal/harmonic support rather than isolated spectral spikes.",
        "cqt": "CQT Peaks uses a Constant-Q Transform with music-like logarithmic frequency spacing. It can be useful for pitched material and interval reasoning, but it is slower and heavier than FFT-based extraction.",
    }.get(engine, "The Extraction Engine determines how candidate audio frequencies are detected before stroboscopic mapping.")


def explain_mapping_mode(mode):
    return {
        "none": "None keeps analysed frequencies close to their raw values, except for clipping to RX1 limits. This is useful for diagnostics but often produces frequencies outside the experiential SLS Design range.",
        "rx1_safe": "Fold to RX1 octave-folds candidates into the broad RX1-supported range. It is device-safe but not necessarily tuned to the SLS/alpha design space.",
        "sls_fold": "Fold to SLS octave-folds candidates into the main stroboscopic design range. This preserves musical octave relationships while making the values usable as flicker frequencies.",
        "sls_center": "Fold Near 11.56 Hz chooses an octave placement near a central alpha-like target. This is a good default when the goal is stable SLS-like entrainment rather than raw audio-frequency following.",
        "alpha_array": "Fold to Alpha compresses candidates into a narrower alpha-like range. This is useful when the visual experience should remain closer to alpha-band stimulation.",
    }.get(mode, "Frequency Mapping determines how raw audio-derived frequencies are folded into a stroboscopic range.")


def explain_amplitude_norm(method):
    return {
        "global_linear": "Global Linear divides all amplitudes by the strongest detected amplitude in the full file. It preserves global loudness contrast, but one extreme spike can make the rest of the sequence look quiet.",
        "per_voice_linear": "Per-Voice Linear scales each extracted voice independently. This keeps secondary voices visible even if they are quieter than the dominant voice.",
        "percentile_clip": "Percentile Clipped uses a high percentile rather than the absolute maximum as the scaling reference. This reduces the influence of brief transients or clipping spikes.",
        "log_global": "Log-Compressed Global strongly compresses amplitude differences. It is useful when you want quieter material to remain visually present without making loud sections overwhelmingly bright.",
        "sqrt_global": "SqRt-Compressed Global gently compresses amplitude differences. It is less aggressive than log compression but still boosts quieter sections relative to linear scaling.",
    }.get(method, "Amplitude Normalization controls how spectral strength becomes a 0?100 control signal.")


def explain_duty_method(method):
    return {
        "fixed": "Fixed Duty keeps the flicker on/off proportion constant. This is the simplest and most controlled setting.",
        "amplitude": "Amplitude Mapped makes duty cycle follow the audio amplitude envelope. This is close to the usual beginner / chiptune-style approach where amplitude drives the visual envelope.",
        "inverted_amplitude": "Inverted Amplitude makes strong audio sections produce lower duty cycle and quiet sections produce higher duty cycle. This can create a more negative-space or contrast-inverted feel.",
        "wide_at_high_amp": "Wider Duty : Higher Amplitude maps stronger audio toward a larger ON proportion, using the selected Duty Floor and Duty Ceiling.",
        "narrow_at_high_amp": "Narrower Duty : Higher Amplitude maps stronger audio toward a smaller ON proportion, which can make loud moments sharper or more punctate.",
        "occupancy_proxy": "Spectral Occupancy Proxy estimates how much energy surrounds the selected candidate frequency. Broader local energy can widen or alter duty cycle depending on the floor/ceiling settings.",
        "harmonic_band_proxy": "Harmonic-Band Proxy estimates energy around harmonic partials of the selected candidate. It can make duty cycle respond to tonal/harmonic support rather than raw amplitude alone.",
        "amp_gated_occupancy": "Amplitude-Gated Occupancy uses the occupancy estimate only when amplitude exceeds the selected gate. Below that threshold, duty returns toward the floor.",
    }.get(method, "Duty Cycle Calculation controls the percentage of each flicker cycle spent ON.")


def explain_luminance_method(method):
    return {
        "amplitude": "Amplitude Mapped makes luminance follow the audio amplitude envelope. This matches the standard simple pipeline: stronger audio becomes brighter stimulation.",
        "sqrt_amplitude": "SqRt-Shaped Amplitude brightens quieter material while retaining some loudness contrast.",
        "log_amplitude": "Log-Shaped Amplitude strongly compresses amplitude differences, creating a more even visual brightness contour.",
        "threshold_gated": "Threshold-Gated Amplitude suppresses luminance below the selected amplitude gate. This is useful for silence/low-energy rejection.",
        "fixed": "Fixed Luminance keeps brightness constant regardless of audio amplitude. This is useful for isolating frequency/duty effects from brightness effects.",
    }.get(method, "Luminance Calculation controls how audio-derived amplitude becomes RX1 brightness.")


def explain_smoothing(method):
    return {
        "none": "None preserves the raw extracted frame-by-frame signal.",
        "rolling_mean": "Rolling Mean averages neighbouring frames. It reduces jagged changes and produces smoother transitions.",
        "rolling_median": "Rolling Median suppresses isolated spikes while preserving sharper boundaries than a rolling mean.",
    }.get(method, "Output Smoothing controls how much generated curves are softened over time.")


def server(input, output, session):

    def active_snap():
        mode = str(input.time_snap_mode())

        if mode == "auto":
            return recommended_snap_for_duration(float(input.total_duration()))

        try:
            return max(0.1, float(mode))
        except Exception:
            return recommended_snap_for_duration(float(input.total_duration()))

    points_rv = reactive.Value(make_default_points(60))
    selected_point_id = reactive.Value(None)
    add_mode = reactive.Value(False)
    status = reactive.Value("Ready.")
    audio_status_rv = reactive.Value("No audio analyzed yet.")
    audio_df_rv = reactive.Value(None)
    view_start = reactive.Value(0.0)
    view_end = reactive.Value(60.0)

    def current_snap():
        mode = input.time_snap_mode()
        if mode == "auto":
            return recommended_snap_for_duration(float(input.total_duration()))
        return float(mode)

    def current_view():
        td = float(input.total_duration())
        start = max(0.0, min(float(view_start.get()), td))
        end = max(0.0, min(float(view_end.get()), td))
        if end <= start:
            start = 0.0
            end = td
        return start, end

    def cleaned_points():
        return ensure_boundary_points(points_rv.get(), float(input.total_duration()), current_snap())




    @output
    @render.ui
    def current_configuration_ui():
        chiptune_recipe = (
            "To replicate Chiptune 3.DX, use 1 voice, FFT Harmonic Salience, Fold Near 11.56 Hz / "
            "SLS Centre mapping, Snap to Musical Grid enabled, Global Linear amplitude scaling, "
            "Amplitude Mapped luminance, Spectral Occupancy / harmonic-masked duty, and Export "
            "Motion Mode set to Hold Each Step. In this pipeline, FFT peak candidates are identified "
            "and ranked by harmonic support across partials; a single stable voice is selected with "
            "continuity handling; the selected frequency is octave-folded into the SLS range near "
            "11.56 Hz; luminance follows the audio amplitude envelope; and duty cycle is derived "
            "from local/harmonic spectral occupancy rather than being simply fixed or directly "
            "amplitude-mapped."
        )

        try:
            engine = input.audio_engine()
            voices = int(input.audio_channels())
            step = float(input.audio_step())
            mapping = input.audio_mapping()
            amp_norm = input.audio_amp_norm()
            duty_method = input.audio_duty_method()
            lum_method = input.audio_lum_method()
            smooth_method = input.audio_smoothing()
            interval = input.audio_interval()
            direction = input.audio_interval_dir()
            snap = bool(input.audio_snap_music())

            try:
                export_motion = input.export_motion_mode()
            except Exception:
                export_motion = "hold"

            interval_text = "None / Unison" if interval == "None / Unison" else f"{interval} ({direction.title()})"
            export_text = "Hold Each Step" if export_motion == "hold" else "Interpolate Between Points"

            pipeline_summary = (
                f"You are extracting {voices} voice(s) with {option_label(engine, AUDIO_ENGINE_LABELS)}, "
                f"using one analysis frame every {step:g} s. Frequencies are mapped with "
                f"{option_label(mapping, AUDIO_MAPPING_LABELS)} and interval handling is set to {interval_text}. "
                f"Amplitude is scaled with {option_label(amp_norm, AMP_NORM_LABELS)}. Duty cycle is generated with "
                f"{option_label(duty_method, DUTY_LABELS)}, luminance is generated with "
                f"{option_label(lum_method, LUM_LABELS)}, and RX1 export motion is set to {export_text}."
            )

            return ui.TagList(
                ui.div({"class": "readme-warning"}, chiptune_recipe),
                ui.p(pipeline_summary),
                ui.div(
                    {"class": "summary-grid"},
                    ui.div({"class": "summary-tile"},
                        ui.div({"class": "summary-tile-title"}, "Extraction Engine"),
                        ui.div({"class": "summary-tile-value"}, explain_extraction_engine(engine)),
                    ),
                    ui.div({"class": "summary-tile"},
                        ui.div({"class": "summary-tile-title"}, "Voices / Timing"),
                        ui.div({"class": "summary-tile-value"}, f"{voices} voice(s), {step:g} s step duration. Smaller steps track the music more closely; larger steps produce simpler RX1 curves."),
                    ),
                    ui.div({"class": "summary-tile"},
                        ui.div({"class": "summary-tile-title"}, "Frequency Mapping"),
                        ui.div({"class": "summary-tile-value"}, explain_mapping_mode(mapping)),
                    ),
                    ui.div({"class": "summary-tile"},
                        ui.div({"class": "summary-tile-title"}, "Musical Treatment"),
                        ui.div({"class": "summary-tile-value"}, f"Interval: {interval_text}. Musical grid snapping is {'enabled' if snap else 'disabled'}."),
                    ),
                    ui.div({"class": "summary-tile"},
                        ui.div({"class": "summary-tile-title"}, "Amplitude Normalization"),
                        ui.div({"class": "summary-tile-value"}, explain_amplitude_norm(amp_norm)),
                    ),
                    ui.div({"class": "summary-tile"},
                        ui.div({"class": "summary-tile-title"}, "Duty Cycle"),
                        ui.div({"class": "summary-tile-value"}, explain_duty_method(duty_method)),
                    ),
                    ui.div({"class": "summary-tile"},
                        ui.div({"class": "summary-tile-title"}, "Luminance"),
                        ui.div({"class": "summary-tile-value"}, explain_luminance_method(lum_method)),
                    ),
                    ui.div({"class": "summary-tile"},
                        ui.div({"class": "summary-tile-title"}, "Export Motion"),
                        ui.div({"class": "summary-tile-value"}, f"{export_text}. Interpolation is only applied when STP duration is greater than 0.1 s."),
                    ),
                    ui.div({"class": "summary-tile"},
                        ui.div({"class": "summary-tile-title"}, "Smoothing"),
                        ui.div({"class": "summary-tile-value"}, explain_smoothing(smooth_method)),
                    ),
                ),
            )

        except Exception as e:
            return ui.div(
                {"class": "readme-warning"},
                f"Current Configuration Summary could not render yet: {e}"
            )


    @output
    @render.ui
    def control_reference_ui():
        return ui.TagList(
            ui.h5("Purpose of the App"),
            ui.p("This app is an accessible RX1/RoXiva sequence editor and audio-to-strobe scrubber. It lets a user draw or edit four oscillator curves, analyse an uploaded audio file, convert audio-derived features into frequency, duty cycle, and luminance curves, preview the result, and export RX1-compatible text output."),

            ui.h5("Chiptune 3.DX Pipeline"),
            ui.p('To replicate Chiptune 3.DX, use 1 voice, FFT Harmonic Salience, Fold Near 11.56 Hz / SLS Centre mapping, Snap to Musical Grid enabled, Global Linear amplitude scaling, Amplitude Mapped luminance, Spectral Occupancy / harmonic-masked duty, and Export Motion Mode set to Hold Each Step. In this pipeline, FFT peak candidates are identified and ranked by harmonic support across partials; a single stable voice is selected with continuity handling; the selected frequency is octave-folded into the SLS range near 11.56 Hz; luminance follows the audio amplitude envelope; and duty cycle is derived from local/harmonic spectral occupancy rather than being simply fixed or directly amplitude-mapped.'),

            ui.h5("Extraction Engine"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("FFT Peaks: "), "The fastest and most transparent option. The audio is divided into short windows, each window is transformed into a frequency spectrum, and the strongest peaks are selected as candidate voices. This is the recommended first-choice mode."),
                ui.tags.li(ui.tags.strong("FFT Harmonic Salience: "), "Starts with FFT peaks but scores them using harmonic support. A frequency whose harmonics are also present can be ranked above a strong but isolated/noisy peak. Useful for more tonal material."),
                ui.tags.li(ui.tags.strong("CQT Peaks: "), "Uses the Constant-Q Transform, which has musical/logarithmic frequency spacing. This can better reflect pitched music, but it is slower and more computationally expensive."),
            ),

            ui.h5("Voices to Extract"),
            ui.p("One voice produces a single dominant trajectory and is easiest to interpret. Two voices can separate a lead and accompaniment. Four voices can fill OSC1?OSC4 independently, but may produce busy sequences that need smoothing or manual editing."),

            ui.h5("Audio Step Duration"),
            ui.p("This controls the time spacing between analysis frames. A 0.1 s step follows fast changes but creates many points. A 0.2?0.5 s step is often a good demonstration range. Larger values create simpler, slower-moving curves."),

            ui.h5("FFT Size, Band Low, Band High, and Peak Threshold"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("FFT Size: "), "Larger FFT sizes give better frequency resolution but are slower and may smear time changes. Smaller values are faster and more responsive."),
                ui.tags.li(ui.tags.strong("Band Low / Band High: "), "These define the audio frequency range searched for candidate peaks. Narrowing the band can stop bass rumble or high-frequency noise from dominating."),
                ui.tags.li(ui.tags.strong("Peak Threshold: "), "A relative threshold for candidate detection. Higher values keep only stronger peaks; lower values allow more candidates and more unstable movement."),
            ),

            ui.h5("Frequency Mapping"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("None: "), "Keeps extracted frequencies close to their raw analysed values. Useful mainly for diagnostics."),
                ui.tags.li(ui.tags.strong("Fold to RX1: "), "Octave-folds candidates into the broad RX1-safe frequency range."),
                ui.tags.li(ui.tags.strong("Fold to SLS: "), "Octave-folds candidates into the main stroboscopic design range."),
                ui.tags.li(ui.tags.strong("Fold Near 11.56 Hz: "), "Chooses an octave placement close to a central alpha-like target. This is often a good SLS default."),
                ui.tags.li(ui.tags.strong("Fold to Alpha: "), "Keeps candidates closer to a narrower alpha-like range."),
            ),

            ui.h5("Snap to Musical Grid"),
            ui.p("When enabled, mapped frequencies can be nudged toward nearby pitch-class-aware values if they are close enough. This helps preserve musical relationships after octave folding."),

            ui.h5("Interval Transposition"),
            ui.p("Interval Transposition shifts extracted candidates by musical intervals such as perfect fifth, tritone, or octave. This allows unison, consonant, and dissonant audiovisual mappings to be generated from the same source audio."),

            ui.h5("Amplitude Normalization"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Global Linear: "), "The whole file shares one amplitude scale. Preserves global contrast but can be affected by single loud spikes."),
                ui.tags.li(ui.tags.strong("Per-Voice Linear: "), "Each voice gets its own scale, making quieter voices easier to see."),
                ui.tags.li(ui.tags.strong("Percentile Clipped: "), "Uses a robust percentile so rare spikes do not flatten the rest of the sequence."),
                ui.tags.li(ui.tags.strong("Log-Compressed Global: "), "Strong compression; quiet material becomes more visible."),
                ui.tags.li(ui.tags.strong("SqRt-Compressed Global: "), "Gentler compression; useful when linear is too contrasty but log is too flat."),
            ),

            ui.h5("Duty Cycle Calculation"),
            ui.p("Duty cycle is the percentage of each flicker cycle that is ON. It changes the sharpness and temporal texture of the flicker independently of brightness."),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Fixed Duty: "), "Constant on/off proportion. Best for controlled tests."),
                ui.tags.li(ui.tags.strong("Amplitude Mapped: "), "Duty cycle follows the amplitude envelope."),
                ui.tags.li(ui.tags.strong("Inverted Amplitude: "), "Duty cycle decreases as amplitude increases."),
                ui.tags.li(ui.tags.strong("Wider Duty : Higher Amplitude: "), "Loud sections move toward the Duty Ceiling."),
                ui.tags.li(ui.tags.strong("Narrower Duty : Higher Amplitude: "), "Loud sections move toward the Duty Floor."),
                ui.tags.li(ui.tags.strong("Spectral Occupancy Proxy: "), "Duty responds to local spectral spread around the selected candidate."),
                ui.tags.li(ui.tags.strong("Harmonic-Band Proxy: "), "Duty responds to energy around harmonic partials."),
                ui.tags.li(ui.tags.strong("Amplitude-Gated Occupancy: "), "Occupancy only affects duty when amplitude exceeds the gate."),
            ),

            ui.h5("Luminance Calculation"),
            ui.p("Luminance controls visual brightness. In the usual audio-to-strobe pipeline, luminance is commonly mapped from amplitude."),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Amplitude Mapped: "), "Brightness follows audio amplitude directly."),
                ui.tags.li(ui.tags.strong("SqRt-Shaped Amplitude: "), "Boosts quieter sections while preserving some contrast."),
                ui.tags.li(ui.tags.strong("Log-Shaped Amplitude: "), "Strongly compresses loudness differences."),
                ui.tags.li(ui.tags.strong("Threshold-Gated Amplitude: "), "Suppresses low-amplitude sections below a chosen threshold."),
                ui.tags.li(ui.tags.strong("Fixed Luminance: "), "Brightness stays constant regardless of audio amplitude."),
            ),

            ui.h5("Output Smoothing"),
            ui.p("Smoothing is applied after extraction and mapping. Rolling Mean produces gentle transitions; Rolling Median removes isolated spikes while preserving sharper changes. Avoid over-smoothing if precise rhythmic alignment matters."),

            ui.h5("Overlay and Apply"),
            ui.p("Overlay options display audio-derived traces on top of the editable curves. Apply buttons convert the analysed audio traces into actual OSC control points, after which they can be manually edited like any other curve."),

            ui.h5("Line Style Convention"),
            ui.div(
                ui.tags.span({"class": "readme-pill"}, "Frequency = Solid"),
                ui.tags.span({"class": "readme-pill"}, "Duty Cycle = Dashed"),
                ui.tags.span({"class": "readme-pill"}, "Luminance = Dotted"),
                ui.tags.span({"class": "readme-pill"}, "Amplitude = Dash-Dot"),
            ),
        )

    @output
    @render.text
    def snap_text():
        rec = recommended_snap_for_duration(float(input.total_duration()))
        active = active_snap()

        try:
            mode = input.export_motion_mode()
        except Exception:
            mode = "hold"

        if mode == "interpolate" and active > 0.1:
            motion = "Interpolate Between Points"
        elif mode == "interpolate" and active <= 0.1:
            motion = "Hold Each Step (0.1 s cannot interpolate)"
        else:
            motion = "Hold Each Step"

        return f"Recommended Snap for {float(input.total_duration()):.1f} s is {rec:.1f} s. Active Snap: {active:.1f} s. Export Motion: {motion}."

    @output
    @render.text
    def zoom_text():
        s, e = current_view()
        return f"Viewing {s:.1f}-{e:.1f} s"

    @reactive.effect
    @reactive.event(input.zoom_reset)
    def _():
        view_start.set(0.0)
        view_end.set(float(input.total_duration()))
        status.set("Showing full sequence.")

    @reactive.effect
    @reactive.event(input.zoom_in)
    def _():
        td = float(input.total_duration())
        s, e = current_view()
        centre = (s + e) / 2
        width = max((e - s) * 0.5, 1.0)
        view_start.set(max(0.0, centre - width / 2))
        view_end.set(min(td, centre + width / 2))
        status.set(f"Zoomed in: {view_start.get():.1f}-{view_end.get():.1f} s.")

    @reactive.effect
    @reactive.event(input.zoom_out)
    def _():
        td = float(input.total_duration())
        s, e = current_view()
        centre = (s + e) / 2
        width = min((e - s) * 2, td)
        view_start.set(max(0.0, centre - width / 2))
        view_end.set(min(td, centre + width / 2))
        if view_start.get() <= 0 and view_end.get() >= td:
            view_start.set(0.0)
            view_end.set(td)
        status.set(f"Zoomed out: {view_start.get():.1f}-{view_end.get():.1f} s.")

    @reactive.effect
    @reactive.event(input.pan_left)
    def _():
        td = float(input.total_duration())
        s, e = current_view()
        width = e - s
        shift = width * 0.35
        ns = max(0.0, s - shift)
        ne = min(td, ns + width)
        view_start.set(ns)
        view_end.set(ne)
        status.set(f"Panned left: {ns:.1f}-{ne:.1f} s.")

    @reactive.effect
    @reactive.event(input.pan_right)
    def _():
        td = float(input.total_duration())
        s, e = current_view()
        width = e - s
        shift = width * 0.35
        ne = min(td, e + shift)
        ns = max(0.0, ne - width)
        view_start.set(ns)
        view_end.set(ne)
        status.set(f"Panned right: {ns:.1f}-{ne:.1f} s.")

    @reactive.effect
    @reactive.event(input.total_duration)
    def _():
        td = float(input.total_duration())
        points_rv.set(ensure_boundary_points(points_rv.get(), td, current_snap()))
        if view_end.get() > td:
            view_end.set(td)
        status.set(f"Total duration set to {td:.1f} s.")

    @reactive.effect
    @reactive.event(input.time_snap_mode)
    def _():
        points_rv.set(ensure_boundary_points(points_rv.get(), float(input.total_duration()), current_snap()))
        status.set(f"Time snap set to {current_snap():.1f} s.")

    @output
    @render.ui
    def svg_editor():
        s, e = current_view()
        html = make_svg_editor(
            points=points_rv.get(),
            selected_osc=input.selected_osc(),
            selected_param=input.selected_param(),
            total_duration=float(input.total_duration()),
            snap=current_snap(),
            view_start=s,
            view_end=e,
            freq_view_mode=input.freq_view_mode(),
            add_mode=add_mode.get(),
            audio_df=audio_df_rv.get(),
            overlay_audio_freq=bool(input.overlay_audio_freq()),
            overlay_audio_duty=bool(input.overlay_audio_duty()),
            overlay_audio_lum=bool(input.overlay_audio_lum()),
            overlay_voice=int(input.overlay_voice()),
        )
        return ui.HTML(html)

    @output
    @render.text
    def mouse_text():
        pos = input.mouse_pos()
        if pos is None:
            return "Move over plot to see x/y."

        if input.selected_param() == "freq":
            return f"x = {pos['t']:.1f} s | y = {pos['value']:.2f} Hz"
        if input.selected_param() == "duty":
            return f"x = {pos['t']:.1f} s | y = {int(round(pos['value']))} % duty"
        return f"x = {pos['t']:.1f} s | y = {int(round(pos['value']))} % luminance"

    @output
    @render.text
    def status_text():
        return status.get()

    @output
    @render.text
    def line_count_text():
        n_lines = estimate_stp_line_count(points_rv.get(), float(input.total_duration()), current_snap())
        if n_lines > MAX_STP_LINES:
            return f"{n_lines} / {MAX_STP_LINES} lines: TOO MANY"
        return f"{n_lines} / {MAX_STP_LINES} lines"


    @output
    @render.text
    def audio_status():
        return audio_status_rv.get()

    @output
    @render.ui
    def audio_player_ui():
        audio_path, audio_name = get_uploaded_audio_path(input)

        if audio_path is None:
            return ui.HTML(
                '<div class="small-note">Upload an audio file to enable browser playback.</div>'
            )

        try:
            uri = file_to_data_uri(audio_path, audio_name)

            return ui.HTML(
                f'''
                <audio id="uploaded_audio_player" controls preload="metadata" style="width:100%;">
                  <source src="{uri}">
                  Your browser does not support the audio element.
                </audio>
                '''
            )

        except Exception as e:
            return ui.HTML(
                f'<div class="small-note">Could not prepare audio player: {e}</div>'
            )

    @reactive.effect
    @reactive.event(input.point_select)
    def _():
        event = input.point_select()
        if event is None:
            return

        pid = int(event["point_id"])
        selected_point_id.set(pid)

        pts = points_rv.get()
        row = pts[pts["point_id"] == pid]

        if len(row) == 1:
            r = row.iloc[0]
            ui.update_numeric("selected_t", value=round(float(r["t"]), 1))
            ui.update_numeric("selected_value", value=float(r["value"]))
            status.set(f"Selected point: {r['osc']} {r['param']} at {r['t']:.1f} s, {r['value']:.2f}.")

    @reactive.effect
    @reactive.event(input.point_drag_final)
    def _():
        event = input.point_drag_final()
        if event is None:
            return

        pid = int(event["point_id"])
        t_new = clamp_time(event["t"], float(input.total_duration()), current_snap())
        v_new = clamp_value(input.selected_param(), event["value"])

        pts = points_rv.get().copy()
        idx = pts.index[pts["point_id"] == pid].tolist()

        if len(idx) == 1:
            pts.loc[idx[0], "t"] = t_new
            pts.loc[idx[0], "value"] = v_new

            pts = clean_points_for_rx1(pts, float(input.total_duration()), current_snap())
            points_rv.set(pts)

            curve = selected_curve(pts, input.selected_osc(), input.selected_param()).copy()
            curve["dist"] = np.sqrt((curve["t"] - t_new) ** 2 + (curve["value"] - float(v_new)) ** 2)
            nearest = curve.sort_values("dist").iloc[0]

            selected_point_id.set(int(nearest["point_id"]))

            ui.update_numeric("selected_t", value=round(float(nearest["t"]), 1))
            ui.update_numeric("selected_value", value=float(nearest["value"]))

            status.set(f"Moved point to {nearest['t']:.1f} s, {nearest['value']:.2f}.")

    @reactive.effect
    @reactive.event(input.add_point_mode)
    def _():
        add_mode.set(True)
        status.set("Add mode on: click the plot to add a point.")

    @reactive.effect
    @reactive.event(input.svg_click)
    def _():
        if not add_mode.get():
            return

        event = input.svg_click()
        if event is None:
            return

        t_new = clamp_time(event["t"], float(input.total_duration()), current_snap())
        v_new = clamp_value(input.selected_param(), event["value"])

        new_row = pd.DataFrame(
            {
                "osc": [input.selected_osc()],
                "param": [input.selected_param()],
                "t": [t_new],
                "value": [v_new],
            }
        )

        pts = points_rv.get()
        pts = pts[
            ~(
                (pts["osc"] == input.selected_osc())
                & (pts["param"] == input.selected_param())
                & (np.abs(pts["t"] - t_new) < 1e-8)
            )
        ]

        pts = pd.concat([pts, new_row], ignore_index=True)
        pts = clean_points_for_rx1(pts, float(input.total_duration()), current_snap())
        points_rv.set(pts)

        curve = selected_curve(pts, input.selected_osc(), input.selected_param()).copy()
        curve["dist"] = np.sqrt((curve["t"] - t_new) ** 2 + (curve["value"] - float(v_new)) ** 2)
        nearest = curve.sort_values("dist").iloc[0]

        selected_point_id.set(int(nearest["point_id"]))
        ui.update_numeric("selected_t", value=round(float(nearest["t"]), 1))
        ui.update_numeric("selected_value", value=float(nearest["value"]))

        add_mode.set(False)
        status.set(f"Added point at {t_new:.1f} s, {v_new:.2f}.")

    @reactive.effect
    @reactive.event(input.apply_selected_values)
    def _():
        pid = selected_point_id.get()
        if pid is None:
            status.set("No point selected.")
            return

        pts = points_rv.get().copy()
        idx = pts.index[pts["point_id"] == pid].tolist()

        if len(idx) != 1:
            status.set("Selected point could not be found.")
            return

        t_new = clamp_time(input.selected_t(), float(input.total_duration()), current_snap())
        v_new = clamp_value(input.selected_param(), input.selected_value())

        pts.loc[idx[0], "t"] = t_new
        pts.loc[idx[0], "value"] = v_new

        pts = clean_points_for_rx1(pts, float(input.total_duration()), current_snap())
        points_rv.set(pts)

        status.set(f"Applied typed values: {t_new:.1f} s, {v_new:.2f}.")

    @reactive.effect
    @reactive.event(input.delete_point)
    def _():
        pid = selected_point_id.get()
        if pid is None:
            status.set("No point selected.")
            return

        pts = points_rv.get()
        row = pts[pts["point_id"] == pid]

        if len(row) != 1:
            status.set("Selected point not found.")
            return

        osc = row.iloc[0]["osc"]
        param = row.iloc[0]["param"]
        curve = selected_curve(pts, osc, param)

        if len(curve) <= 2:
            status.set("Cannot delete: each curve needs at least two points.")
            return

        pts = pts[pts["point_id"] != pid]
        pts = clean_points_for_rx1(pts, float(input.total_duration()), current_snap())
        points_rv.set(pts)
        selected_point_id.set(None)
        status.set("Deleted selected point.")

    @reactive.effect
    @reactive.event(input.copy_curve_all)
    def _():
        pts = points_rv.get()
        curve = selected_curve(pts, input.selected_osc(), input.selected_param())

        if input.selected_osc() == "SUN":
            status.set("SUN/halogen is a global curve and cannot be copied to OSC1-OSC4.")
            return

        copied = []
        for osc in OSC_NAMES:
            c = curve.copy()
            c["osc"] = osc
            copied.append(c[["osc", "param", "t", "value"]])

        pts = pts[pts["param"] != input.selected_param()]
        pts = pd.concat([pts] + copied, ignore_index=True)
        pts = clean_points_for_rx1(pts, float(input.total_duration()), current_snap())
        points_rv.set(pts)

        status.set(f"Copied {input.selected_param()} curve to all oscillators.")

    @reactive.effect
    @reactive.event(input.reset_curve)
    def _():
        default = 60 if input.selected_param() == "freq" else 50 if input.selected_param() == "duty" else 0

        new_curve = pd.DataFrame(
            {
                "osc": [input.selected_osc(), input.selected_osc()],
                "param": [input.selected_param(), input.selected_param()],
                "t": [0.0, float(input.total_duration())],
                "value": [default, default],
            }
        )

        pts = points_rv.get()
        pts = pts[~((pts["osc"] == input.selected_osc()) & (pts["param"] == input.selected_param()))]
        pts = pd.concat([pts, new_curve], ignore_index=True)
        pts = clean_points_for_rx1(pts, float(input.total_duration()), current_snap())
        points_rv.set(pts)
        selected_point_id.set(None)
        status.set("Reset selected curve.")

    @reactive.effect
    @reactive.event(input.analyze_audio)
    def _():
        files = input.audio_file()
        if not files:
            audio_status_rv.set("No audio file uploaded.")
            return

        try:
            path = files[0]["datapath"]

            audio_status_rv.set("Audio Analysis started.")

            with ui.Progress(min=0, max=1) as p:
                def progress_update(value, message, detail=""):
                    p.set(value=value, message=message, detail=detail)
                    audio_status_rv.set(f"{message} | {detail}")

                df = analyze_audio_file(
                    audio_path=path,
                    engine=input.audio_engine(),
                    n_voices=int(input.audio_channels()),
                    step_duration=float(input.audio_step()),
                    n_fft=int(input.audio_n_fft()),
                    band_lo=float(input.audio_band_lo()),
                    band_hi=float(input.audio_band_hi()),
                    peak_rel_height=float(input.audio_peak_rel()),
                    mapping_mode=input.audio_mapping(),
                    snap_music=bool(input.audio_snap_music()),
                    interval_name=input.audio_interval(),
                    interval_direction=input.audio_interval_dir(),
                    amplitude_norm=input.audio_amp_norm(),
                    amp_percentile=float(input.audio_amp_percentile()),
                    duty_method=input.audio_duty_method(),
                    fixed_duty=int(input.audio_fixed_duty()),
                    duty_floor=float(input.audio_duty_floor()),
                    duty_ceiling=float(input.audio_duty_ceiling()),
                    duty_amp_gate=float(input.audio_duty_amp_gate()),
                    luminance_method=input.audio_lum_method(),
                    fixed_lum=int(input.audio_fixed_lum()),
                    lum_floor=float(input.audio_lum_floor()),
                    lum_ceiling=float(input.audio_lum_ceiling()),
                    lum_amp_gate=float(input.audio_lum_amp_gate()),
                    occupancy_bw_cents=float(input.audio_occ_bw()),
                    smoothing_method=input.audio_smoothing(),
                    smoothing_window=int(input.audio_smooth_window()),
                    progress=progress_update,
                )

            audio_df_rv.set(df)

            duration = float(df["Time"].max()) if len(df) else float(input.total_duration())
            ui.update_numeric("total_duration", value=max(duration, 1.0))
            view_start.set(0.0)
            view_end.set(max(duration, 1.0))

            audio_status_rv.set(f"Audio analyzed: {len(df)} frames, duration ~{duration:.1f} s.")

        except Exception as e:
            audio_status_rv.set(f"Audio Analysis failed: {e}")

    def apply_audio_to_editor(include_freq=True, include_duty=True, include_lum=True, all_oscs=False):
        audio_df = audio_df_rv.get()

        if audio_df is None or len(audio_df) == 0:
            status.set("No audio analysis available.")
            return

        base_pts = points_rv.get()
        new_parts = []

        if all_oscs:
            for i, osc in enumerate(OSC_NAMES, start=1):
                voice = min(i, 4)
                new_parts.append(
                    audio_trace_to_points(
                        audio_df,
                        osc=osc,
                        voice=voice,
                        include_freq=True,
                        include_duty=True,
                        include_lum=True,
                    )
                )
            remove_params = PARAM_KEYS
            remove_oscs = OSC_NAMES

        else:
            osc = input.selected_osc()
            voice = int(input.overlay_voice())
            new_parts.append(
                audio_trace_to_points(
                    audio_df,
                    osc=osc,
                    voice=voice,
                    include_freq=include_freq,
                    include_duty=include_duty,
                    include_lum=include_lum,
                )
            )
            remove_params = []
            if include_freq:
                remove_params.append("freq")
            if include_duty:
                remove_params.append("duty")
            if include_lum:
                remove_params.append("lum")
            remove_oscs = [osc]

        keep = base_pts.copy()
        for osc in remove_oscs:
            for param in remove_params:
                keep = keep[~((keep["osc"] == osc) & (keep["param"] == param))]

        combined = pd.concat([keep] + new_parts, ignore_index=True)
        td = max(float(input.total_duration()), float(audio_df["Time"].max()))
        combined = clean_points_for_rx1(combined, td, current_snap())
        points_rv.set(combined)

        status.set("Applied audio trace(s) to editor curves.")

    @reactive.effect
    @reactive.event(input.apply_audio_selected_freq)
    def _():
        apply_audio_to_editor(include_freq=True, include_duty=False, include_lum=False, all_oscs=False)

    @reactive.effect
    @reactive.event(input.apply_audio_selected_all)
    def _():
        apply_audio_to_editor(include_freq=True, include_duty=True, include_lum=True, all_oscs=False)

    @reactive.effect
    @reactive.event(input.apply_audio_all_oscs)
    def _():
        apply_audio_to_editor(all_oscs=True)

    @output
    @render.plot
    def audio_plot():
        return plot_audio_scrubber(
            audio_df_rv.get(),
            input.freq_view_mode(),
            playhead_time=None
        )

    @output
    @render.plot
    def audio_waveform_spectrogram_plot():
        audio_path, _ = get_uploaded_audio_path(input)
        return plot_audio_waveform_spectrogram(
            audio_path,
            playhead_time=None
        )

    @output
    @render.plot
    def preview_plot():
        return plot_preview(
            cleaned_points(),
            float(input.total_duration()),
            input.freq_view_mode(),
            playhead_time=None
        )

    @render.download(filename=lambda: f"{input.sequence_name()}.txt")
    def download_txt():
        lines = make_stp_lines(points_rv.get(), float(input.total_duration()), current_snap())
        yield "\r\n".join(lines)

    @render.download(filename=lambda: f"{input.sequence_name()}_project.json")
    def download_project():
        project = {
            "sequence_name": input.sequence_name(),
            "total_duration": float(input.total_duration()),
            "time_snap_mode": input.time_snap_mode(),
            "active_snap": current_snap(),
            "freq_view_mode": input.freq_view_mode(),
            "max_stp_lines": MAX_STP_LINES,
            "rx1_constraints": {
                "time_min_step": TIME_MIN_STEP,
                "freq_min": FREQ_MIN,
                "freq_max": FREQ_MAX,
                "duty_min": DUTY_MIN,
                "duty_max": DUTY_MAX,
                "lum_min": LUM_MIN,
                "lum_max": LUM_MAX,
            },
            "points": cleaned_points()[["osc", "param", "t", "value"]].to_dict(orient="records"),
        }
        yield json.dumps(project, indent=2)

    @render.download(filename=lambda: f"{input.sequence_name()}_preview.png")
    def download_png():
        fig = plot_preview(cleaned_points(), float(input.total_duration()), input.freq_view_mode())
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, dpi=300, bbox_inches="tight")
            plt.close(fig)
            with open(tmp.name, "rb") as f:
                data = f.read()
        yield data

    @render.download(filename=lambda: f"{input.sequence_name()}_audio_analysis.csv")
    def download_audio_csv():
        df = audio_df_rv.get()
        if df is None:
            yield "No audio analysis available.\n"
        else:
            yield df.to_csv(index=False)


    @render.download(filename=lambda: f"d_{sanitize_filename_stem(input.sequence_name(), max_len=18)}_various.lscf")
    def download_lucio_lscf():
        files = input.lucio_template()
        if not files:
            yield b"Upload d_spcwkdc1020904022_various.lscf first."
            return

        template_path = files[0]["datapath"]
        template_data = Path(template_path).read_bytes()
        lscf_bytes, _ = lucio_build_lscf_from_points(
            points=points_rv.get(),
            total_duration=float(input.total_duration()),
            template_data=template_data,
            control_step_seconds=float(input.lucio_control_step()),
        )
        yield lscf_bytes

    @render.download(filename=lambda: f"{sanitize_filename_stem(input.sequence_name(), max_len=18)}_lucio_studio_debug.csv")
    def download_lucio_debug():
        files = input.lucio_template()
        if not files:
            yield "Upload d_spcwkdc1020904022_various.lscf first.\n"
            return

        template_path = files[0]["datapath"]
        template_data = Path(template_path).read_bytes()
        _, debug_df = lucio_build_lscf_from_points(
            points=points_rv.get(),
            total_duration=float(input.total_duration()),
            template_data=template_data,
            control_step_seconds=float(input.lucio_control_step()),
        )
        yield debug_df.to_csv(index=False)

    @reactive.effect
    @reactive.event(input.upload_project)
    def _():
        files = input.upload_project()
        if not files:
            return

        try:
            path = files[0]["datapath"]
            with open(path, "r", encoding="utf-8") as f:
                project = json.load(f)

            required = {"sequence_name", "total_duration", "points"}
            if not required.issubset(project.keys()):
                status.set("Invalid project file.")
                return

            pts = pd.DataFrame(project["points"])
            if not {"osc", "param", "t", "value"}.issubset(pts.columns):
                status.set("Project file is missing required point columns.")
                return

            duration = float(project["total_duration"])

            pts = pts[pts["osc"].isin(EDITOR_OSC_NAMES) & pts["param"].isin(PARAM_KEYS)].copy()
            pts["t"] = pts["t"].astype(float)
            pts["value"] = pts["value"].astype(float)

            ui.update_text("sequence_name", value=project.get("sequence_name", "loaded_sequence"))
            ui.update_numeric("total_duration", value=duration)
            ui.update_select("time_snap_mode", selected=project.get("time_snap_mode", "auto"))
            ui.update_select("freq_view_mode", selected=project.get("freq_view_mode", "sls"))

            pts = ensure_boundary_points(pts, duration, recommended_snap_for_duration(duration))
            points_rv.set(pts)

            view_start.set(0.0)
            view_end.set(duration)
            selected_point_id.set(None)
            add_mode.set(False)

            status.set("Loaded editable JSON project.")

        except Exception as e:
            status.set(f"Failed to load project: {e}")


app = App(app_ui, server)
