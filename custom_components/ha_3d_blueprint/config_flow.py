"""Config flow for HA 3D Blueprint."""
import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, FlowResult
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    EntitySelector,
    EntitySelectorConfig,
)


from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class Blueprint3DConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for HA 3D Blueprint."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Here you would normally validate the user_input
            # For now, we just accept it.
            _LOGGER.info("Creating new HA 3D Blueprint config entry")
            return self.async_create_entry(title="3D Blueprint", data=user_input)

        # This is the form the user will see.
        data_schema = vol.Schema(
            {
                vol.Required(
                    "square_footage",
                    description="Total square footage of the building",
                    default=2000,
                ): int,
                vol.Required(
                    "num_floors",
                    description="Number of floors in the building",
                    default=1,
                ): int,
            }
        )

        return self.async_show_form(
            step_id="user", data_schema=data_schema, errors=errors
        )