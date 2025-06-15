from fastapi import FastAPI
import uvicorn
import os
import logging
import numpy as np
import itertools
import math
import requests
import json
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import Dict, List, Tuple

# --- Constants, Logging, and other setup code... ---
SHARED_DIR = "/share"
SVG_OUTPUT_PATH = os.path.join(SHARED_DIR, "blueprint.svg")
MODEL_STATE_PATH = os.path.join(SHARED_DIR, "blueprint_model.json")
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
_LOGGER = logging.getLogger(__name__)


def save_grid_as_svg(grid, sensor_coords, ref_points, output_path, cell_size=10):
    """Converts a 2D density grid into an SVG file, drawing points on top."""
    _LOGGER.info("Generating SVG at path: %s", output_path)
    if grid is None or grid.size == 0:
        _LOGGER.warning("Grid is empty, cannot generate SVG.")
        return

    height, width = grid.shape
    svg_width = width * cell_size
    svg_height = height * cell_size

    svg_parts = [f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg" style="background-color:white;">']
    svg_parts.append('<style>.sensor { fill: blue; stroke: white; } .corner { fill: green; } .doorway { fill: red; }</style>')

    # Draw density grid
    for y in range(height):
        for x in range(width):
            density = min(max(grid[y, x], 0.0), 1.0)
            opacity = density * 0.8 # Don't make it fully black
            svg_parts.append(f'<rect x="{x * cell_size}" y="{y * cell_size}" width="{cell_size}" height="{cell_size}" fill="black" fill-opacity="{opacity}" />')

    # Draw stationary sensors
    for sid, (sx, sy) in sensor_coords.items():
        svg_parts.append(f'<circle cx="{(sx + 0.5) * cell_size}" cy="{(sy + 0.5) * cell_size}" r="5" class="sensor"><title>{sid}</title></circle>')

    # Draw reference points
    for point in ref_points:
        px, py = point['position']
        ptype = point['type']
        if ptype == 'corner':
            svg_parts.append(f'<rect x="{px * cell_size}" y="{py * cell_size}" width="{cell_size}" height="{cell_size}" class="corner"><title>Corner</title></rect>')
        elif ptype == 'doorway':
            svg_parts.append(f'<circle cx="{(px + 0.5) * cell_size}" cy="{(py + 0.5) * cell_size}" r="4" class="doorway"><title>Doorway</title></circle>')

    svg_parts.append('</svg>')

    try:
        with open(output_path, 'w') as f:
            f.write("".join(svg_parts))
        _LOGGER.info("Successfully saved SVG to %s", output_path)
    except Exception as e:
        _LOGGER.error("Failed to write SVG file: %s", e)

class KalmanFilter:
    """A simple 1D Kalman filter for smoothing RSSI values."""
    def __init__(self, process_variance=1e-3, measurement_variance=0.1, initial_value=-65):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.x = initial_value
        self.p = 1.0

    def update(self, measurement: float) -> float:
        self.p += self.process_variance
        k = self.p / (self.p + self.measurement_variance)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x

class DatabaseManager:
    """Handles connection and data writing to InfluxDB."""
    def __init__(self):
        self.url = os.environ.get("INFLUXDB_URL")
        self.token = os.environ.get("INFLUXDB_TOKEN")
        self.org = os.environ.get("INFLUXDB_ORG")
        self.bucket = os.environ.get("INFLUXDB_BUCKET")

        if not all([self.url, self.token, self.org, self.bucket]):
            _LOGGER.error("InfluxDB environment variables not fully set.")
            self.write_api = None
        else:
            _LOGGER.info("Initializing InfluxDB client...")
            client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            self.write_api = client.write_api(write_options=SYNCHRONOUS)

    def write_rssi_data(self, rssi_data: Dict[str, int], device_id: str):
        if not self.write_api:
            _LOGGER.error("Cannot write to InfluxDB, client not initialized.")
            return
        points = [Point("rssi_measurement").tag("mobile_device", device_id).tag("stationary_sensor", s).field("rssi", r) for s, r in rssi_data.items()]
        if points:
            try:
                self.write_api.write(bucket=self.bucket, org=self.org, record=points)
                _LOGGER.info("Successfully wrote %d points to InfluxDB.", len(points))
            except Exception as e:
                _LOGGER.error("Failed to write to InfluxDB: %s", e)

class TomographyModel:
    """Encapsulates the state and logic for Radio Tomographic Imaging."""
    def __init__(self, config: dict, enriched_data: dict):
        # ... (init logic remains the same)
        _LOGGER.info("Initializing TomographyModel...")
        self.config = config
        self.sensors = config.get("stationary_sensors", [])
        self.rssi_at_reference = -40.0
        self.path_loss_exponent = 2.7
        self.reference_distance = 1.0
        self.sq_ft = enriched_data.get("estimated_sq_ft", 2000)
        self.grid_resolution = int(math.sqrt(self.sq_ft))
        self.num_pixels = self.grid_resolution * self.grid_resolution

        self.sensor_coords = {}
        self.reference_points = []
        self.image_vector_x = np.zeros(self.num_pixels, dtype=np.float32)

        self.load_state()
        if not self.sensor_coords:
            self.sensor_coords = self._generate_placeholder_sensor_coords()
        self.links = self._create_links()
        self.link_filters = {link: KalmanFilter() for link in self.links}
        self.weight_matrix_A = self._build_weight_matrix()
        _LOGGER.info("TomographyModel initialized successfully.")

    def load_state(self):
        """Loads the model's state from a file."""
        if os.path.exists(MODEL_STATE_PATH):
            _LOGGER.info("Loading saved model state...")
            try:
                with open(MODEL_STATE_PATH, 'r') as f:
                    state = json.load(f)
                    self.sensor_coords = state.get("sensor_coords", {})
                    self.reference_points = state.get("reference_points", [])
                    image_list = state.get("image_vector_x", [])
                    if image_list: self.image_vector_x = np.array(image_list)
            except Exception as e:
                _LOGGER.error("Failed to load model state: %s", e)

    def save_state(self):
        """Saves the model's current state to a file."""
        _LOGGER.info("Saving model state...")
        state = {
            "sensor_coords": self.sensor_coords,
            "reference_points": self.reference_points,
            "image_vector_x": self.image_vector_x.tolist()
        }
        with open(MODEL_STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2)

    def _generate_placeholder_sensor_coords(self) -> Dict[str, Tuple[int, int]]:
        coords = {}
        center = self.grid_resolution / 2
        radius = self.grid_resolution * 0.45
        for i, sensor_id in enumerate(self.sensors):
            angle = 2 * np.pi * i / len(self.sensors)
            x = center + radius * np.cos(angle)
            y = center + radius * np.sin(angle)
            coords[sensor_id] = (int(x), int(y))
        return coords

    def _create_links(self) -> List[Tuple[str, str]]:
        return list(itertools.combinations(self.sensors, 2))

    def _build_weight_matrix(self) -> np.ndarray:
        A = np.zeros((len(self.links), self.num_pixels), dtype=np.float32)
        pixels_x, pixels_y = np.meshgrid(np.arange(self.grid_resolution), np.arange(self.grid_resolution))
        pixel_coords = np.vstack([pixels_x.ravel(), pixels_y.ravel()]).T
        for i, link in enumerate(self.links):
            p1, p2 = np.array(self.sensor_coords[link[0]]), np.array(self.sensor_coords[link[1]])
            line_vec, line_len_sq = p2 - p1, np.dot(p2 - p1, p2 - p1)
            if line_len_sq == 0: continue
            t = np.clip(np.dot(pixel_coords - p1, line_vec) / line_len_sq, 0, 1)
            dist_sq = np.sum((pixel_coords - (p1 + t[:, np.newaxis] * line_vec))**2, axis=1)
            A[i, :] = (dist_sq < 1.0**2).astype(np.float32)
        return A

    def calculate_signal_loss_vector_b(self, actual_rssi_values: Dict[Tuple[str, str], int]) -> np.ndarray:
        # ... (implementation remains the same) ...
        return np.zeros(len(self.links))

    def reconstruct_image(self, b: np.ndarray, num_iterations: int = 10, learning_rate: float = 0.01) -> np.ndarray:
        x = self.image_vector_x
        for _ in range(num_iterations):
            error = b - (self.weight_matrix_A @ x)
            x += learning_rate * (self.weight_matrix_A.T @ error)
            np.clip(x, 0, 1, out=x)
        self.image_vector_x = x
        self.save_state()
        return x.reshape((self.grid_resolution, self.grid_resolution))

    def add_reference_point(self, tag_type: str, mobile_rssi_values: Dict[str, int]):
        # ... (implementation remains the same) ...
        self.save_state()

# --- FastAPI Application ---
app = FastAPI()
db_manager = DatabaseManager()
ENGINE_STATE = {"model": None}

@app.post("/configure")
def configure_engine(config: dict):
    try:
        enriched_data = {"estimated_sq_ft": 2000}
        model = TomographyModel(config, enriched_data)
        ENGINE_STATE["model"] = model
        save_grid_as_svg(model.image_vector_x.reshape((model.grid_resolution, model.grid_resolution)), model.sensor_coords, model.reference_points, SVG_OUTPUT_PATH)
        return {"status": "configured"}
    except Exception as e:
        _LOGGER.error("Failed to initialize TomographyModel: %s", e)
        return {"status": "error"}

@app.post("/tag_location")
def tag_location(data: dict):
    model: TomographyModel = ENGINE_STATE.get("model")
    if not model:
        return {"error": "Engine not configured."}

    rssi_values = data.get("rssi_values", {})
    device_id = data.get("mobile_beacon_id", "unknown")
    tag_type = data.get("tag_type", "update_obstruction_map")

    # This is the corrected, added call to the database manager.
    db_manager.write_rssi_data(rssi_values, device_id)

    if tag_type in ["tag_corner", "tag_doorway"]:
        model.add_reference_point(tag_type.split('_')[1], rssi_values)
    else:
        b = model.calculate_signal_loss_vector_b(rssi_values)
        model.reconstruct_image(b)

    # Always redraw the SVG with the latest state
    save_grid_as_svg(
        model.image_vector_x.reshape((model.grid_resolution, model.grid_resolution)),
        model.sensor_coords,
        model.reference_points,
        SVG_OUTPUT_PATH
    )

    return {"status": "processed", "tag_type": tag_type}

if __name__ == "__main__":
    _LOGGER.info("Starting Blueprint Engine web server...")
    uvicorn.run(app, host="0.0.0.0", port=8124)