# arduino-app-launcher

A long-lived supervisor that runs inside one shared `python-apps-base` container (`/run.sh serve`). It imports the heavy libraries once, then starts each Arduino app as a forked child on request, so a start skips the interpreter startup and the library imports: on a VENTUNO Q a camera app logs "App started" after 1.2 s instead of 3.0 s.

Run it with `arduino-app-launcher` or `python -m arduino.app_tools.app_launcher`. It stops on SIGTERM or SIGINT, after stopping its apps; during the preload it just exits, so a stop is never lost even as PID 1.

With a command, the same executable is a client of the control socket that prints the JSON response (`client.py`, stdlib only, a reference for other clients):

```
arduino-app-launcher [--socket PATH] ping | status | start <app> [--env K=V ...] [--mode replace] | stop [<app>]
```

## Processes

- **Supervisor** (`supervisor.py`): one asyncio loop in one thread. Owns the control socket, the table of running apps keyed by app folder, the `/app` link, the per-app logs (`applog.py`) and the TCP ingress (`ingress.py`).
- **App child** (`child.py`, stdlib only): forked from the supervisor, it becomes the app the way `python main.py` would run in the classic container.

## Control socket

A Unix socket at `APP_LAUNCHER_SOCKET`, mode 0660. One JSON object per line, one request per connection, then the response and the connection closes. Lifecycle requests (`start`, `stop`) run one at a time; `status` and `ping` answer immediately.

| Request | Response |
|---|---|
| `{"cmd":"start","app":"<abs folder>","env":{...},"mode":"replace"}` | `{"ok":true,"pid":N,"started_at":"<iso>","warm":true,"runner":"python","timings":{"stop_ms":..,"fork_ms":..}}` |
| `{"cmd":"stop"}` or `{"cmd":"stop","app":"<abs folder>"}` | `{"ok":true,"stopped":[{"app":..,"pid":..,"exit_code":..,"signal":..}]}` |
| `{"cmd":"status"}` | `{"ok":true,"children":[{"app","pid","state","started_at","http_port","runner","warm"}],"preload":{"modules":[..],"failed":[..]},"version":..,"uptime_s":..}` |
| `{"cmd":"ping"}` | `{"ok":true}` |

`start` with `replace` stops every running app first, then forks the new one. `mode` `add` is refused with `unsupported`: one app at a time for now. Stopping an app that is not running is not an error. A stop sends SIGTERM, waits the grace period, then SIGKILLs the app's process group; whatever the app leaves in its group is killed once it exits, as in a container whose main process ends.

Errors are `{"ok":false,"error":{"code":..,"message":..}}` with `code` one of `bad_request` (malformed request, relative app path), `not_found` (no app folder or no `python/main.py`), `unsupported` (`mode` `add`), `busy` (the launcher is shutting down), `internal` (fork refused or failed, unexpected errors).

## Environment

| Variable | Default | |
|---|---|---|
| `APP_LAUNCHER_PRELOAD` | `arduino.app_utils,arduino.app_peripherals.camera,arduino.app_peripherals.microphone,arduino.app_bricks.web_ui` | Modules imported at startup, one by one; a failing one is logged and reported by `status` |
| `APP_LAUNCHER_SOCKET` | `/run/arduino-app-cli/app-launcher.sock` | Control socket; a stale one is replaced |
| `APP_LAUNCHER_STOP_TIMEOUT_S` | `5` | Grace period between SIGTERM and SIGKILL, the CLI's container stop timeout |
| `APP_LAUNCHER_PORTS` | `7000` | Public ports forwarded to the running app at `port + 10000` on 127.0.0.1 |
| `APP_LAUNCHER_LINK` | `/home/app/.launcher/current` | Symlink pointed at the running app; skipped when its folder does not exist |

## The app child

Before any app code runs, the child:

1. becomes a session and process group leader;
2. points every inherited fd above 2 at `/dev/null` and its stdout and stderr at the log pipe;
3. gets back the signal dispositions the supervisor had after the preload, which are those of a fresh `python main.py`;
4. gets the supervisor's environment with the request's `env` on top, then `ARDUINO_APP_HTTP_PORT` (the internal port) and `PYTHONUNBUFFERED=1`, and `<app>/bricks` on `PYTHONPATH` when it exists;
5. moves to the app folder, sets `sys.argv` and `sys.path` as a fresh interpreter would;
6. runs `python/main.py` as a permanent `__main__` module: a fresh module with the dunders of a script, not `runpy`, so the app's threads and atexit handlers keep seeing it. Uncaught exceptions go through `sys.excepthook` without the launcher frames, with exit code 1; `SystemExit` is honoured and the interpreter exits normally.

Two kinds of apps are not run warm: the child execs instead, with the same environment, cwd and process group, and `start` reports `"warm":false`.

- An app with dependencies of its own: its venv `<app>/.cache/.venv` holds a `*.dist-info` or a `*.pth` other than the `_virtualenv.pth` of an empty `uv venv`, or it has a non-blank `python/requirements.txt` or wheels in `python-libraries`. The child execs the venv interpreter on `main.py`, with `VIRTUAL_ENV` set and its `bin` first on `PATH`. An app that declares dependencies but has no venv yet runs warm.
- A Streamlit app, whose `app.yaml` mentions `arduino:streamlit_ui` as run.sh checks: the child execs `<python> -m streamlit run --server.port <internal port> <app>/python/main.py`, with the venv interpreter when the app has dependencies of its own, else the launcher's. `start` reports `"runner":"streamlit"`.

The output of each app goes to `<app>/.cache/logs/app.log`, rotated at 5 MB into a single `app.log.1`. The supervisor's own lines go to its stdout, the container log.

## Fork rules

- The supervisor is single-threaded: no executors, no subprocess helpers, numeric addresses only so asyncio never resolves names in a thread. A fork is refused while `threading.active_count() > 1`, with the names of the threads alive. The "Ready" line also counts the OS threads on Linux, which shows native threads that `threading` cannot see.
- The fork happens between two runs of the event loop, never inside a callback, so the child never unwinds through asyncio.
- Signals are blocked across the fork and the GC is frozen in the child, so no supervisor finalizer ever runs inside the app. The child never closes the inherited event loop: on Linux the epoll instance is shared with the parent.
- Importing the library installs no signal handler; the app's `App.run()` installs its own SIGTERM handler as in the classic container.

## The /app limitation

The library and the apps hardcode `/app` paths (`app.yaml`, `/app/assets`, `/app/data`, `/app/certs`...). In the image `/app` is a root-owned symlink to `/home/app/.launcher/current`, and `WORKDIR` is `/home/app`. The classic mode bind-mounts the app folder on `/app`, which lands on `current`; in the serve mode the supervisor atomically replaces `current` with a link to the app folder on every `start`. So `/app` works for one app at a time. Running several apps at once needs the library to use `APP_HOME` instead.
