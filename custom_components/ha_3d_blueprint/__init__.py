import logging

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS = ["camera", "button"]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA 3D Blueprint from a config entry."""
    _LOGGER.info("Setting up HA 3D Blueprint for entry %s", entry.title)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {"config": entry.data}

    await async_configure_addon(hass, entry)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_configure_addon(hass: HomeAssistant, entry: ConfigEntry):
    """Send the configuration data to the Blueprint Engine add-on."""
    _LOGGER.info("Sending configuration to Blueprint Engine Add-on...")

    config_data = entry.data
    addon_url = "http://blueprint_engine.local.hass.io:8124/configure"
    session = async_get_clientsession(hass)

    try:
        async with session.post(addon_url, json=config_data) as response:
            if response.status == 200:
                _LOGGER.info("Successfully configured Blueprint Engine Add-on.")
            else:
                _LOGGER.error(
                    "Failed to configure add-on. Status: %s, Response: %s",
                    response.status,
                    await response.text(),
                )
    except Exception as e:
        _LOGGER.error("Error communicating with Blueprint Engine Add-on: %s", e)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok
