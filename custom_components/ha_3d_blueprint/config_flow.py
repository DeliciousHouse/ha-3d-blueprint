import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, FlowResult
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
)
from homeassistant.helpers import area_registry as ar
from homeassistant.const import CONF_API_KEY

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

class Blueprint3DConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HA 3D Blueprint."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step. We now automatically discover data."""
        self.lat = self.hass.config.latitude
        self.lon = self.hass.config.longitude

        area_registry = ar.async_get(self.hass)
        self.areas = {
            area.id: {"name": area.name, "floor": area.floor_id}
            for area in area_registry.async_list_areas()
        }
        return await self.async_step_sensors()

    async def async_step_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the step where the user selects sensors and provides an optional API key."""
        errors: dict[str, str] = {}
        if user_input is not None:
            combined_data = {
                "latitude": self.lat,
                "longitude": self.lon,
                "areas": self.areas,
                "stationary_sensors": user_input.get("stationary_sensors", []),
                "mobile_beacon_sensor": user_input.get("mobile_beacon_sensor"),
                "api_key": user_input.get(CONF_API_KEY, ""),
            }
            return self.async_create_entry(title="3D Blueprint", data=combined_data)

        data_schema = vol.Schema(
            {
                vol.Required("stationary_sensors"): EntitySelector(
                    EntitySelectorConfig(domain="sensor", multiple=True)
                ),
                vol.Required("mobile_beacon_sensor"): EntitySelector(
                    EntitySelectorConfig(domain="sensor")
                ),
                vol.Optional(CONF_API_KEY): str,
            }
        )

        description = (
            "Please select your stationary Bluetooth sensor entities and the "
            "single 'BLE Transmitter' sensor entity from your mobile phone."
        )

        return self.async_show_form(
            step_id="sensors",
            data_schema=data_schema,
            errors=errors,
            description_placeholders={"description": description}
        )