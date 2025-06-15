import logging
from datetime import datetime, timezone

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
        """Handle the button press by sending a timestamped event."""
        tag_type = self.entity_description.key
        _LOGGER.info("'%s' button pressed!", self.name)

        addon_url = "http://blueprint_engine.local.hass.io:8124/tag_location"
        session = async_get_clientsession(self.hass)

        # The button's only job is to provide the "what" and "when".
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag_type": tag_type,
            "mobile_beacon_id": self.entry.data.get("mobile_beacon_sensor")
        }

        _LOGGER.debug("Sending payload to add-on: %s", payload)
        try:
            async with session.post(addon_url, json=payload) as response:
                if response.status == 200:
                    _LOGGER.info("Successfully sent '%s' tag to Blueprint Engine.", tag_type)
                else:
                    _LOGGER.error("Failed to send tag: %s", response.status)
        except Exception as e:
            _LOGGER.error("Error communicating with add-on: %s", e)