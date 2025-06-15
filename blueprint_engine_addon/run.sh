#!/usr/bin/with-contenv bashio
# ==============================================================================
# Home Assistant Community Add-on: Blueprint Engine
#
# This script is executed when the add-on is started.
# ==============================================================================

bashio::log.info "Starting the 3D Blueprint Engine..."

# Export the log level configuration option to an environment variable
# This lets our Python script know how verbose to be.
export LOG_LEVEL=$(bashio::config 'log_level')
bashio::log.info "Log level is set to: ${LOG_LEVEL}"

# The main command to start the Python application.
# The "-u" flag ensures that Python output is sent straight to the
# add-on logs without being buffered, which is good for debugging.
python3 -u /usr/bin/engine.py

bashio::log.info "3D Blueprint Engine has stopped."