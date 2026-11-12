"""Sensor noise models — radar, AIS, EO/IR.

Each sensor model takes a vessel's true state and produces a noisy observation
that mimics the error characteristics of the real sensor modality. These
observations are what the perception module of the pipeline actually consumes.

Sensor characteristics
----------------------
- **Radar**: bearing error (Gaussian, ~1°), range error (Gaussian, ~30 m),
  sea clutter (Poisson-distributed false returns), missed detections.
- **AIS**: position error (Gaussian, ~10 m), speed error, message dropouts
  (5% of messages), update rate ~3-10 s.
- **EO/IR**: pixel-level Gaussian noise, occlusion, brightness variation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .vessels import Vessel, VesselState


# ───────────────────────────────────────────────────────────
# Observation types
# ───────────────────────────────────────────────────────────

@dataclass
class RadarObservation:
    """A single radar observation of a vessel."""
    time_s: float
    bearing_deg: float              # bearing from radar (0 = north)
    range_m: float                   # range from radar
    range_rate_ms: float             # radial velocity (closing rate)
    target_id: Optional[str] = None  # if associated (typically unknown in raw radar)
    is_clutter: bool = False         # True if this is a clutter return, not a vessel
    detection_confidence: float = 1.0


@dataclass
class AISMessage:
    """A single AIS message."""
    time_s: float
    mmsi: str                        # Maritime Mobile Service Identity
    lat: float
    lon: float
    heading_deg: float
    speed_kn: float
    nav_status: str = "under_way"   # navigation status


@dataclass
class EOIRFrame:
    """A single EO/IR image frame."""
    time_s: float
    image: np.ndarray                # shape (H, W, C)
    detections: list[dict] = field(default_factory=list)
    # Each detection: {bbox_xyxy, label, true_track_id}
    # The image contains pixel-level noise; detections are ground-truth-ish.


# ───────────────────────────────────────────────────────────
# Radar sensor
# ───────────────────────────────────────────────────────────

class RadarSensor:
    """Marine radar sensor with bearing/range error and sea clutter.

    Parameters
    ----------
    position : tuple[float, float]
        Radar position in metres (north, east) relative to scenario origin.
    bearing_error_std_deg : float
        Standard deviation of bearing measurement error.
    range_error_std_m : float
        Standard deviation of range measurement error.
    detection_probability : float
        Probability of detecting a vessel per scan (otherwise missed detection).
    clutter_rate : float
        Average number of clutter returns per scan (Poisson).
    max_range_m : float
        Maximum detection range.
    """

    def __init__(
        self,
        position_m: tuple[float, float] = (0.0, 0.0),
        bearing_error_std_deg: float = 1.0,
        range_error_std_m: float = 30.0,
        detection_probability: float = 0.95,
        clutter_rate: float = 3.0,
        max_range_m: float = 24_000.0,
        seed: int = 0,
    ):
        self.position_m = position_m
        self.bearing_error_std_deg = bearing_error_std_deg
        self.range_error_std_m = range_error_std_m
        self.detection_probability = detection_probability
        self.clutter_rate = clutter_rate
        self.max_range_m = max_range_m
        self.rng = np.random.default_rng(seed)

    def observe(self, vessels: list[Vessel], time_s: float) -> list[RadarObservation]:
        """Observe all vessels within radar range at the given time."""
        observations: list[RadarObservation] = []
        radar_n, radar_e = self.position_m

        for vessel in vessels:
            state = vessel.current_state
            vn, ve = vessel.position_m

            # True range and bearing
            dn = vn - radar_n
            de = ve - radar_e
            true_range = float(np.hypot(dn, de))
            if true_range > self.max_range_m:
                continue  # out of range

            true_bearing = float(np.degrees(np.arctan2(de, dn))) % 360

            # Detection probability check
            if self.rng.random() > self.detection_probability:
                continue  # missed detection

            # Add Gaussian noise to bearing and range
            obs_bearing = (true_bearing + self.rng.normal(0, self.bearing_error_std_deg)) % 360
            obs_range = max(0.0, true_range + self.rng.normal(0, self.range_error_std_m))

            # Range rate (radial velocity) — projected relative velocity
            own_vel_n = state.speed_kn * 0.5144 * np.cos(np.radians(state.heading_deg))
            own_vel_e = state.speed_kn * 0.5144 * np.sin(np.radians(state.heading_deg))
            # Relative velocity to radar (radar assumed stationary)
            radial_vel = (own_vel_n * dn + own_vel_e * de) / max(true_range, 1.0)

            observations.append(RadarObservation(
                time_s=time_s,
                bearing_deg=obs_bearing,
                range_m=obs_range,
                range_rate_ms=float(radial_vel),
                target_id=vessel.vessel_id,
                is_clutter=False,
                detection_confidence=float(0.7 + 0.3 * (1 - true_range / self.max_range_m)),
            ))

        # Generate sea clutter returns (Poisson-distributed)
        n_clutter = self.rng.poisson(self.clutter_rate)
        for _ in range(n_clutter):
            clutter_bearing = self.rng.uniform(0, 360)
            clutter_range = self.rng.uniform(100, self.max_range_m)
            observations.append(RadarObservation(
                time_s=time_s,
                bearing_deg=float(clutter_bearing),
                range_m=float(clutter_range),
                range_rate_ms=float(self.rng.normal(0, 5)),
                target_id=None,
                is_clutter=True,
                detection_confidence=float(self.rng.uniform(0.2, 0.5)),
            ))

        return observations


# ───────────────────────────────────────────────────────────
# AIS sensor
# ───────────────────────────────────────────────────────────

class AISSensor:
    """AIS transponder receiver.

    Models the AIS broadcast characteristics:
    - Position error ~10 m (Class A)
    - Speed/heading error
    - Message dropouts (~5%)
    - Update rate (Class A: 2-10 s depending on speed/status)
    """

    def __init__(
        self,
        position_error_std_m: float = 10.0,
        speed_error_std_kn: float = 0.1,
        heading_error_std_deg: float = 0.5,
        dropout_probability: float = 0.05,
        update_interval_s: float = 3.0,
        seed: int = 0,
    ):
        self.position_error_std_m = position_error_std_m
        self.speed_error_std_kn = speed_error_std_kn
        self.heading_error_std_deg = heading_error_std_deg
        self.dropout_probability = dropout_probability
        self.update_interval_s = update_interval_s
        self.rng = np.random.default_rng(seed)
        self._last_broadcast_time: dict[str, float] = {}

    def observe(self, vessels: list[Vessel], time_s: float) -> list[AISMessage]:
        """Receive AIS messages from vessels due to broadcast."""
        messages: list[AISMessage] = []
        METERS_PER_DEG_LAT = 111_320.0
        ref_lat = 13.0  # default reference latitude

        for vessel in vessels:
            last_time = self._last_broadcast_time.get(vessel.vessel_id, -1e9)
            if time_s - last_time < self.update_interval_s:
                continue  # not yet due to broadcast

            # Dropout check
            if self.rng.random() < self.dropout_probability:
                self._last_broadcast_time[vessel.vessel_id] = time_s
                continue  # message dropped

            self._last_broadcast_time[vessel.vessel_id] = time_s
            state = vessel.current_state

            # Add noise to position (in metres, then convert to degrees)
            pos_error_n = self.rng.normal(0, self.position_error_std_m)
            pos_error_e = self.rng.normal(0, self.position_error_std_m)
            noisy_lat = state.lat + pos_error_n / METERS_PER_DEG_LAT
            noisy_lon = state.lon + pos_error_e / (METERS_PER_DEG_LAT * np.cos(np.radians(ref_lat)))

            # Add noise to heading and speed
            noisy_heading = (state.heading_deg + self.rng.normal(0, self.heading_error_std_deg)) % 360
            noisy_speed = max(0.0, state.speed_kn + self.rng.normal(0, self.speed_error_std_kn))

            messages.append(AISMessage(
                time_s=time_s,
                mmsi=f"MMSI{vessel.vessel_id}",
                lat=noisy_lat,
                lon=noisy_lon,
                heading_deg=noisy_heading,
                speed_kn=noisy_speed,
                nav_status="under_way",
            ))

        return messages


# ───────────────────────────────────────────────────────────
# EO/IR sensor
# ───────────────────────────────────────────────────────────

class EOIRSensor:
    """Electro-optical / infrared camera.

    Generates synthetic image frames with vessel detections. The image is
    intentionally small (64x64) to keep storage low while still demonstrating
    the perception pipeline.

    Parameters
    ----------
    image_size : tuple[int, int, int]
        Image shape (height, width, channels).
    noise_std : float
        Standard deviation of pixel-level Gaussian noise.
    fov_deg : float
        Field of view (horizontal).
    """

    def __init__(
        self,
        image_size: tuple[int, int, int] = (64, 64, 3),
        noise_std: float = 0.05,
        fov_deg: float = 60.0,
        occlusion_probability: float = 0.05,
        seed: int = 0,
    ):
        self.image_size = image_size
        self.noise_std = noise_std
        self.fov_deg = fov_deg
        self.occlusion_probability = occlusion_probability
        self.rng = np.random.default_rng(seed)

    def observe(self, vessels: list[Vessel], time_s: float) -> EOIRFrame:
        """Generate a single EO/IR frame observing the given vessels."""
        h, w, c = self.image_size
        # Sky / sea background
        image = np.zeros((h, w, c), dtype=np.float32)
        # Sky (top half) — lighter blue
        image[:h//2] = 0.6
        # Sea (bottom half) — darker
        image[h//2:] = 0.3

        # Add Gaussian pixel noise
        image += self.rng.normal(0, self.noise_std, image.shape).astype(np.float32)
        image = np.clip(image, 0, 1)

        detections: list[dict] = []
        for vessel in vessels:
            state = vessel.current_state
            # Project vessel position to image coordinates
            # (simplified — assumes vessels are below the horizon)
            vn, ve = vessel.position_m
            range_m = float(np.hypot(vn, ve))
            if range_m > 5000:
                continue  # out of frame

            # Horizontal pixel based on bearing
            bearing_rad = np.arctan2(ve, vn)
            x_pixel = int(w / 2 + (bearing_rad / np.radians(self.fov_deg / 2)) * w / 2)
            # Vertical pixel based on range (closer = lower in image)
            y_pixel = int(h * 0.7 - 0.4 * (1 - range_m / 5000) * h)

            if 0 <= x_pixel < w and 0 <= y_pixel < h:
                # Check occlusion
                if self.rng.random() < self.occlusion_probability:
                    continue  # occluded

                # Draw vessel as a small blob
                blob_size = max(2, int(8 * (1 - range_m / 5000)))
                x0 = max(0, x_pixel - blob_size // 2)
                x1 = min(w, x_pixel + blob_size // 2)
                y0 = max(0, y_pixel - blob_size // 2)
                y1 = min(h, y_pixel + blob_size // 2)
                # Vessel brightness depends on class colour (grayscale approximation)
                brightness = 0.85
                image[y0:y1, x0:x1] = brightness

                detections.append({
                    "bbox_xyxy": (float(x0), float(y0), float(x1), float(y1)),
                    "label": vessel.vessel_class.value,
                    "true_track_id": vessel.vessel_id,
                    "range_m": range_m,
                })

        return EOIRFrame(time_s=time_s, image=image, detections=detections)
