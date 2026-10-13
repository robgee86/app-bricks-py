# WebUI - HTML Brick

This Brick is a simplified, embeddable web server designed for hosting frontend applications and exposing APIs or WebSocket communication channels.

## Overview

The WebUI - HTML Brick allows you to:

- Serve an HTML+JavaScript web interface (e.g., dashboards, control panels, SPAs)
- Expose REST APIs to be consumed by your frontend or third-party clients
- Communicate in real time with browsers using WebSockets
- Integrate with other bricks to display data or control devices over the network

Once started, your application will be accessible via a web browser at `http://<device-ip>:<port>` (default port 7000).

## Features

- Serves static HTML, CSS, and JavaScript files
- Supports RESTful API endpoints using FastAPI-style handlers
- Customizable routes and handlers
- Simple configuration for port and root directory
- Lightweight and suitable for embedded devices
- Logging of HTTP requests and errors

## Code example and usage

```python
from arduino.app_utils import App
from arduino.app_bricks.web_ui import WebUI

# Initialize the Web UI server
web_ui = WebUI()

# Add a simple REST API endpoint
web_ui.expose_api("GET", "/hello", lambda: {"message": "Hello, world!"})

# Send a welcome message over WebSocket to each client that connects
web_ui.on_connect(lambda sid: web_ui.send_message("hello", {"message": f"Hello, {sid}!"}))

# Start the app: this runs the server and blocks until the app is stopped
App.run()

# The server now serves the static files from the assets folder of the app and responds to /hello requests
```

The server is started by `App.run()` — you don't need to call any method on `web_ui` to start it. API routes are registered under `api_path_prefix`, which defaults to `""` (so the endpoint above is `/hello`).

WebSocket messages can only be delivered while the server is running: call `send_message()` from a callback (like `on_connect` or `on_message` above) or from another brick's loop, not before `App.run()`.

## Configuration

`WebUI()` accepts the following optional constructor parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `addr` | `"0.0.0.0"` | Server bind address |
| `port` | `7000` | Server port shown in the URLs (`0` picks a free port) |
| `ui_path_prefix` | `""` | URL prefix for UI routes |
| `api_path_prefix` | `""` | URL prefix for API routes |
| `assets_dir_path` | `assets` in the app folder, usually `"/app/assets"` | Static assets directory; must contain an `index.html` if present, otherwise `start()` raises `RuntimeError` |
| `certs_dir_path` | `certs` in the app folder, usually `"/app/certs"` | TLS certificates directory |
| `use_tls` | `False` | Enable TLS/HTTPS |
| `cors_origins` | `"*"` | CORS allowed origins (`"*"`, comma-separated list, or `""` to disable) |

The `ARDUINO_APP_PORT_<port>` environment variable, named after the configured `port` (e.g. `ARDUINO_APP_PORT_7000`), when set to a port between 1 and 65535, overrides the port the server binds to, while `url`, `local_url` and the logged URLs keep showing `port`. This lets a proxy in front of the app serve the public port and forward to the internal one, as the app launcher does. Invalid values are logged and ignored.

## Main methods

- `expose_api(method, path, function)`: registers a REST endpoint (FastAPI-style handler).
- `expose_camera(path, camera, jpeg_quality=80)`: exposes a camera stream in MJPEG format, consumable from an `<img>` tag.
- `on_connect(callback)` / `on_disconnect(callback)`: WebSocket connection callbacks; the callback receives the client session ID.
- `on_message(message_type, callback)`: handles a WebSocket message type; the callback receives `(sid, data)` and its return value, if any, is sent back as `<message_type>_response`.
- `send_message(message_type, message, room=None)`: emits a WebSocket message to all connected clients (or a specific room).
- `url` / `local_url`: externally/locally addressable URL of the server.

