# arduino-app-launcher

A long-lived supervisor that runs inside the container of one Arduino app, started by `/run.sh serve` of `python-apps-base`. It imports the heavy libraries once, then starts the app as a forked child on request, so a start skips the interpreter startup and the library imports. The container stays up once the app stops, so the next start, a reload after an edit, is a fork rather than a container start.

Run it with `python -m arduino.app_tools.app_launcher` on the interpreter the app runs on; `arduino-app-launcher` is the same program on the system interpreter. It opens its socket first and preloads the libraries (`APP_LAUNCHER_PRELOAD`, a comma separated list of modules) only when idle, half a second after opening and one module between two turns of its loop: the start a fresh container is brought up for forks at once and the app imports what it needs itself, the next start is warm. It stops on SIGTERM or SIGINT, after stopping the app, and exits with 128 plus the signal number as a stopped app does, so a container stopped by the daemon reads as stopped and not as failed; during an import it just exits, so a stop is never lost.

With a command, the same executable is a client of the control socket that prints the JSON response (`client.py`, stdlib only, a reference for other clients):

```
arduino-app-launcher [--socket PATH] ping | status | start <app> [--env K=V ...] | stop
```

## What runs where

The compose project of the app is the one the daemon generates today: the app folder bind-mounted at `/app`, the environment, the ports, the devices and the sidecars of the bricks all declared there. What changes is the command of the main service, `/run.sh serve` instead of `/run.sh`, and a bind mount for the folder holding the control socket. Without the `serve` argument run.sh runs the app as it always did, so a release frozen with the plain entrypoint keeps working, cold.

`run.sh serve` does what a classic start does before running the app, the venv and the installs of the requirements, the wheels and the custom bricks included, then runs the launcher on the interpreter the app runs on, the venv one when the app has a venv, with `APP_DIR` naming the app folder. The launcher serves that folder only, and records the fingerprint of its dependencies as run.sh left them.

The daemon starts the app through the socket, and a start of the app while it runs is a reload: the launcher stops it and forks it again. A stop stops the project as it always did, container included; the container outlives the app only when the app ends on its own, which keeps the start after a crash warm. What the launcher cannot do warm is left to the paths that exist today:

- A change of the app dependencies, the requirements, the wheels, the requirements of a custom brick or a Streamlit brick added, is refused as `stale`: the daemon restarts the container, run.sh installs and the launcher comes back fresh. `ping` and `status` tell `stale` ahead of a start.
- A change of the compose project, a brick with a sidecar, a new port, a device, an environment variable or the image, is the daemon's to see: it recreates the project, as it does today, and the new container gets a launcher of its own.

## Processes

- **PID 1**: the launcher itself when the service has no `init`, where it reaps the orphans and handles the stop; docker-init with `init: true`.
- **Supervisor** (`supervisor.py`): one asyncio loop in one thread. Owns the control socket and the one app process. It listens on no TCP port.
- **App child** (`child.py`, stdlib only): forked from the supervisor, it becomes the app the way `python main.py` would run it.

## Control socket

A Unix socket at `APP_LAUNCHER_SOCKET`, mode 0660. One JSON object per line, one request per connection, then the response and the connection closes. Lifecycle requests (`start`, `stop`) run one at a time; `status` and `ping` answer immediately.

| Request | Response |
|---|---|
| `{"cmd":"start","app":"/app","env":{"K":"V"}}` | `{"ok":true,"pid":N,"started_at":"<iso>","runner":"python","timings":{"stop_ms":..,"fork_ms":..}}` |
| `{"cmd":"stop"}` | `{"ok":true,"stopped":[{"app":..,"pid":..,"exit_code":..,"signal":..}]}` |
| `{"cmd":"status"}` | `{"ok":true,"child":{"app","pid","runner","started_at","stopping"} or null,"last":{"app","pid","exit_code","signal","ended_at"} or null,"stale":bool,"venv":..,"preload":{"modules":[..],"loaded":[..],"failed":[..]},"version":..,"uptime_s":..}` |
| `{"cmd":"ping"}` | `{"ok":true,"stale":bool}` |

`start` takes the absolute app folder and an optional `env`, an object of strings laid over the container environment. A start with an app running stops it first, a reload, and reports the time in `timings.stop_ms`. `stop` stops the app, `stopped` holds zero or one entry; an `app` in the request is ignored. Stopping when nothing runs is not an error. `last` in `status` is how the previous app process ended, which tells an app that crashed from one that was stopped while the container lives on.

Errors are `{"ok":false,"error":{"code":..,"message":..}}` with `code` one of `bad_request` (malformed request, relative app path, `env` not an object of strings, a folder other than the served one), `not_found` (no app folder or no `python/main.py`), `stale` (the dependencies changed since run.sh prepared them, nothing was done, the running app goes on), `busy` (the launcher is shutting down), `internal` (fork refused or failed, unexpected errors).

## Stopping and what the app leaves behind

A stop sends SIGTERM to the app's process group, waits `APP_LAUNCHER_STOP_TIMEOUT_S`, then SIGKILLs it. Once the app is gone, every other process of the container is killed but PID 1 and the supervisor, helpers in sessions of their own included, listed again from `/proc` for the ones forked meanwhile. The same sweep follows an app that exits on its own. The sweep runs only when the supervisor tops a pid namespace of its own, as PID 1 or the child of PID 1: elsewhere, a development machine or a test, the other processes are not the app's and only the process group is killed.

## Environment

The child runs with the container environment updated by the `env` of the start request, then the additions run.sh makes before running an app: `PYTHONUNBUFFERED=1`, `<app>/bricks` on `PYTHONPATH` when it exists, and `VIRTUAL_ENV` with its `bin` first on `PATH` when the launcher runs in the app venv.

| Variable | Default | |
|---|---|---|
| `APP_DIR` | unset | The app folder run.sh prepared, the only one served; unset, any folder is served and its dependencies are fixed at its first start |
| `APP_LAUNCHER_PRELOAD` | `arduino.app_utils`, the camera and microphone backends, `arduino.app_peripherals.speaker`, `arduino.app_bricks.web_ui` | Modules imported when idle, one by one; a failing one is logged and reported by `status`. In the app venv `streamlit.web.bootstrap` is added when the venv has it |
| `APP_LAUNCHER_SOCKET` | `/run/arduino-app-cli/app-launcher.sock` | Control socket; a stale one is replaced |
| `APP_LAUNCHER_STOP_TIMEOUT_S` | `5` | Grace period between SIGTERM and SIGKILL |
| `APP_LAUNCHER_LOG_LEVEL` | `WARNING` | What the launcher logs of its own |

## Logs

The app inherits the container stdout and stderr: what it prints is read with `docker logs`, as in the classic stack. The supervisor writes its own lines to stderr, prefixed with `app-launcher:`, warnings only unless `APP_LAUNCHER_LOG_LEVEL` asks for `info` or `debug`.

## The app child

Before any app code runs, the child:

1. becomes a session and process group leader;
2. points every inherited fd above 2 at `/dev/null`, keeping the container stdin, stdout and stderr;
3. gets back the signal dispositions the supervisor had after the preload, which are those of a fresh `python main.py`, and reseeds NumPy's global generator, which the fork copied from the warm interpreter;
4. moves to the app folder, takes the environment above and sets `sys.argv` and `sys.path` as a fresh interpreter would;
5. installs the startup SIGTERM handler of the library, which a fresh interpreter gets when main.py imports it, so that a stop before `App.run()` exits cleanly;
6. runs `python/main.py` as a permanent `__main__` module: a fresh module with the dunders of a script, not `runpy`, so the app's threads and atexit handlers keep seeing it. Uncaught exceptions go through `sys.excepthook` without the launcher frames, with exit code 1; `SystemExit` is honoured and the interpreter exits normally.

A Streamlit app, whose `app.yaml` mentions `arduino:streamlit_ui` as run.sh checks, is served through `streamlit.web.bootstrap` on port 7000, the port run.sh always gave Streamlit, as `streamlit run` does; `start` reports `"runner":"streamlit"`. run.sh installs Streamlit in the app venv before the launcher starts, so the child finds it in its own interpreter.

## Fork rules

- The supervisor is single-threaded: no executors, no subprocess helpers, numeric addresses only so asyncio never resolves names in a thread. A fork is refused while `threading.active_count() > 1`, with the names of the threads alive. The "Ready" line also counts the OS threads on Linux, which shows native threads that `threading` cannot see.
- The fork happens between two runs of the event loop, never inside a callback, so the child never unwinds through asyncio.
- Signals are blocked across the fork and the GC is frozen in the child, so no supervisor finalizer ever runs inside the app. The child never closes the inherited event loop: on Linux the epoll instance is shared with the parent.
- The supervisor ends as `App.run` ends an app: the exit handlers run, the streams are flushed and the process exits without the teardown of the preloaded modules.
