# HA 3D Blueprint

[![hacs_badge](https://img.shields.io/badge/HACS-Default-orange.svg)](https://github.com/hacs/integration)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced Home Assistant integration and add-on that aims to create a 2D/3D model of your home's layout using Bluetooth RSSI values from your existing sensors.

**This project is under active development and is considered experimental.**

---

## Overview

This project uses the principles of Radio Tomographic Imaging (RTI) to generate a "density map" of your home. By analyzing the signal strength between your stationary Bluetooth devices (like speakers, proxies, etc.), it can infer the location of obstructions like walls.

By using the "Tag" buttons, you can help the system learn by marking the precise locations of corners and doorways, allowing it to build an accurate wireframe of your home's layout over time.

The project consists of two parts that work together:
1.  **The Integration:** Provides the user interface within Home Assistant (buttons, sensors, and the final camera entity to display the blueprint).
2.  **The Add-on:** A powerful backend engine that handles the database, performs the heavy mathematical calculations, and generates the SVG blueprint image.

## Prerequisites

Before installing, you must have the following set up in your Home Assistant:

1.  **InfluxDB Add-on:** This project requires the official [InfluxDB Home Assistant Add-on](https://github.com/home-assistant/addons/blob/master/influxdb/DOCS.md) to be installed and running. You will need to create a database (bucket) for this project to use.
2.  **Bluetooth Sensors:** At least 4-5 stationary Bluetooth-enabled devices that report RSSI values of other devices (e.g., ESPresense proxies, Home Assistant Companion App's BLE Transmitter).
3.  **Mobile Beacon:** A Bluetooth sensor that represents your mobile phone, such as the "BLE Transmitter" sensor from the Home Assistant Companion App.

---

## Installation

### 1. Install the Blueprint Engine Add-on

This is the easiest way to install the add-on.

1.  Navigate to **Settings > Add-ons > Add-on Store**.
2.  Click the three-dots menu in the top right corner and select **"Repositories"**.
3.  Paste the following URL into the box and click **"Add"**:
    ```
    [https://github.com/DeliciousHouse/ha-3d-blueprint](https://github.com/DeliciousHouse/ha-3d-blueprint)
    ```
4.  Close the dialog. Back in the Add-on Store, refresh the page.
5.  The "Blueprint Engine" add-on will now appear. Click it, install it, and configure the InfluxDB connection details in the "Configuration" tab before starting it.

### 2. Install the HA 3D Blueprint Integration

This integration is best installed via the Home Assistant Community Store (HACS). If you do not have HACS installed, please follow the [official HACS installation guide](https://hacs.xyz/docs/setup/download).

1.  **Add as a Custom Repository in HACS:**
    * Go to HACS > Integrations.
    * Click the three-dots menu in the top right and select "Custom Repositories".
    * Paste your GitHub repository URL: `https://github.com/DeliciousHouse/ha-3d-blueprint`
    * Select the category "Integration".
    * Click "Add".
2.  **Install the Integration:**
    * The "HA 3D Blueprint" integration will now appear in your HACS list.
    * Click on it and then click "Download".
    * Restart Home Assistant when prompted.

---

## Configuration

Once the Add-on is running and the Integration is installed, you can add it to Home Assistant.

Click the button below to start the configuration flow:

[![Open your Home Assistant instance and start setting up a new integration.](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=ha_3d_blueprint)

The setup wizard will guide you through selecting your stationary sensors and your mobile beacon sensor.

## Usage

After configuration, three buttons and a camera entity will be created.

1.  Add the `camera.3d_blueprint_live_view` entity to your dashboard to see the generated map.
2.  Add the `button.tag_corner` and `button.tag_doorway` buttons to your mobile phone's dashboard.
3.  As you walk around your house, press the appropriate button when you are physically standing in a corner or doorway to help the system learn your home's layout.
4.  Press the `button.update_obstruction_map` to trigger a new reconstruction of the wall layout based on the latest sensor-to-sensor data.

