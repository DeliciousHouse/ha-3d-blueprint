import logging
from datetime import datetime

from homeassistant.components.button import ButtonEntity, ButtonEntityDescription
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

BUTTON_DESCRIPTIONS = (
    ButtonEntityDescription(key="tag_corner", name="Tag Corner", icon="mdi:vector-square"),
    ButtonEntityDescription(key="tag_doorway", name="Tag Doorway", icon="mdi:door"),
    ButtonEntityDescription(key="update_obstruction_map", name="Update Obstruction Map", icon="mdi:wall"),
)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the button platform."""
    buttons = [BlueprintButton(hass, entry, description) for description in BUTTON_DESCRIPTIONS]
    async_add_entities(buttons)


class BlueprintButton(ButtonEntity):
    """A button to trigger blueprinting actions."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, description: ButtonEntityDescription) -> None:
        """Initialize the button."""
        self.hass = hass
        self.entry = entry
        self.entity_description = description
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": "HA 3D Blueprint Control",
        }

    async def async_press(self) -> None:
        """Handle the button press."""
        tag_type = self.entity_description.key
        _LOGGER.info("'%s' button pressed!", self.name)

        addon_url = "http://blueprint_engine.local.hass.io:8124/tag_location"
        session = async_get_clientsession(self.hass)

        config = self.entry.data
        mobile_beacon_id = config.get("mobile_beacon_sensor")
        stationary_sensors = config.get("stationary_sensors", [])

        if not mobile_beacon_id:
            _LOGGER.error("Mobile Beacon Sensor not configured.")
            return

        mobile_state = self.hass.states.get(mobile_beacon_id)
        if not mobile_state:
            _LOGGER.error("State for mobile beacon '%s' not found.", mobile_beacon_id)
            return

        # The mobile beacon's attributes should contain RSSI values for nearby devices.
        # This is common for ESPresense or the HA Companion App's BLE Transmitter.
        beacons_seen = mobile_state.attributes.get("beacons", {})

        rssi_values = {}
        for sensor_id in stationary_sensors:
            # The sensor_id might be a full entity_id, or just a name.
            # We need to find the corresponding key in the beacons_seen dict.
            # This logic might need adjustment based on the exact format.
            for beacon_name, beacon_data in beacons_seen.items():
                 if sensor_id in beacon_name or beacon_name in sensor_id:
                     rssi_values[sensor_id] = beacon_data.get("rssi")
                     break

        if not rssi_values:
            _LOGGER.warning("Could not find any configured stationary sensors in the mobile beacon's data.")
            return

        payload = {
            "timestamp": datetime.now().isoformat(),
            "tag_type": tag_type,
            "mobile_beacon_id": mobile_beacon_id,
            "rssi_values": rssi_values
        }

        _LOGGER.debug("Sending payload to add-on: %s", payload)
        try:
            async with session.post(addon_url, json=payload) as response:
                if response.status == 200:
                    _LOGGER.info("Successfully sent tag to Blueprint Engine.")
                else:
                    _LOGGER.error("Failed to send tag: %s", response.status)
        except Exception as e:
            _LOGGER.error("Error communicating with add-on: %s", e)