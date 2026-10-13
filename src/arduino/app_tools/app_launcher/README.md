# arduino-app-launcher

A long-lived supervisor that runs inside one shared `python-apps-base` container (`/run.sh serve`, its default), the only way apps run. It imports the heavy libraries once, then starts each Arduino app as a forked child on request, so a start skips the interpreter startup and the library imports: on a VENTUNO Q a camera app logs "App started" after 1.2 s instead of 3.0 s.

Run it with `arduino-app-launcher` or `python -m arduino.app_tools.app_launcher`. It stops on SIGTERM or SIGINT, after stopping its apps; during the preload it just exits, so a stop is never lost even as PID 1.

With a command, the same executable is a client of the control socket that prints the JSON response (`client.py`, stdlib only, a reference for other clients):

```
arduino-app-launcher [--socket PATH] ping | status | start <app> [--env K=V ...] [--port P ...] [--mode replace] | stop [<app>]
```

## Processes

- **Supervisor** (`supervisor.py`): one asyncio loop in one thread. Owns the control socket, the table of running apps keyed by app folder and the per-app logs (`applog.py`). It listens on no TCP port.
- **App child** (`child.py`, stdlib only): forked from the supervisor, it becomes the app the way `python main.py` would run it.

## Control socket

A Unix socket at `APP_LAUNCHER_SOCKET`, mode 0660. One JSON object per line, one request per connection, then the response and the connection closes. Lifecycle requests (`start`, `stop`) run one at a time; `status` and `ping` answer immediately.

| Request | Response |
|---|---|
| `{"cmd":"start","app":"<abs folder>","env":{...},"mode":"replace","ports":[7000]}` | `{"ok":true,"pid":N,"started_at":"<iso>","warm":true,"runner":"python","ports":{"7000":17000},"prepared":false,"timings":{"stop_ms":..,"prepare_ms":..,"fork_ms":..}}` |
| `{"cmd":"stop"}` or `{"cmd":"stop","app":"<abs folder>"}` | `{"ok":true,"stopped":[{"app":..,"pid":..,"exit_code":..,"signal":..}]}` |
| `{"cmd":"status"}` | `{"ok":true,"children":[{"app","pid","state","started_at","ports","runner","warm"}],"preload":{"modules":[..],"failed":[..]},"version":..,"uptime_s":..}` |
| `{"cmd":"ping"}` | `{"ok":true}` |

`ports` lists the public ports of the app, none when absent or empty, see [Ports](#ports). `start` with `replace` stops every running app first, prepares the new one when needed, then forks it. `mode` `add` is refused with `unsupported`: one app at a time for now. Stopping an app that is not running is not an error. A stop sends SIGTERM, waits the grace period, then SIGKILLs the app's process group; whatever the app leaves in its group is killed once it exits, as in a container whose main process ends.

Errors are `{"ok":false,"error":{"code":..,"message":..}}` with `code` one of `bad_request` (malformed request, relative app path, a public port without an internal one), `not_found` (no app folder or no `python/main.py`), `unsupported` (`mode` `add`), `busy` (the launcher is shutting down), `prepare_failed` (`run.sh prepare` failed, was stopped or timed out, with the last lines of its output), `internal` (fork refused or failed, unexpected errors).

## Environment

| Variable | Default | |
|---|---|---|
| `APP_LAUNCHER_PRELOAD` | `arduino.app_utils,arduino.app_peripherals.camera,arduino.app_peripherals.microphone,arduino.app_bricks.web_ui` | Modules imported at startup, one by one; a failing one is logged and reported by `status` |
| `APP_LAUNCHER_SOCKET` | `/run/arduino-app-cli/app-launcher.sock` | Control socket; a stale one is replaced |
| `APP_LAUNCHER_STOP_TIMEOUT_S` | `5` | Grace period between SIGTERM and SIGKILL, the CLI's container stop timeout |
| `APP_LAUNCHER_RUN_SH` | `/run.sh` | The entrypoint whose `prepare` mode installs the app dependencies |
| `APP_LAUNCHER_PREPARE_TIMEOUT_S` | `600` | How long a preparation may take before it is killed; the client waits this long plus a minute for a `start` |

## Ports

The launcher publishes no port: the CLI's ingress on the host network does. Each public port `P` in the `start` request maps to the internal port `P + 10000`, and the app gets `ARDUINO_APP_PORT_<P>=<internal port>` in its environment. A brick serving `P` binds the internal port when the variable is set and `P` otherwise, while the URLs it shows keep `P`. `start` and `status` report the mapping as `ports`, `{"<public>": <internal>}`, with the public ports as string keys as JSON requires.

## The app child

Before any app code runs, the child:

1. becomes a session and process group leader;
2. points every inherited fd above 2 at `/dev/null` and its stdout and stderr at the log pipe;
3. gets back the signal dispositions the supervisor had after the preload, which are those of a fresh `python main.py`;
4. gets the supervisor's environment with the request's `env` on top, then an `ARDUINO_APP_PORT_<P>` per port of the app and `PYTHONUNBUFFERED=1`, and `<app>/bricks` on `PYTHONPATH` when it exists;
5. moves to the app folder, sets `sys.argv` and `sys.path` as a fresh interpreter would;
6. runs `python/main.py` as a permanent `__main__` module: a fresh module with the dunders of a script, not `runpy`, so the app's threads and atexit handlers keep seeing it. Uncaught exceptions go through `sys.excepthook` without the launcher frames, with exit code 1; `SystemExit` is honoured and the interpreter exits normally.

Two kinds of apps are not run warm: the child execs instead, with the same environment, cwd and process group, and `start` reports `"warm":false`.

- An app with dependencies of its own, as the preparation defines them, or whose venv `<app>/.cache/.venv` holds a `*.dist-info` or a `*.pth` other than the `_virtualenv.pth` of an empty `uv venv`. The child execs the venv interpreter on `main.py`, with `VIRTUAL_ENV` set and its `bin` first on `PATH`.
- A Streamlit app, whose `app.yaml` mentions `arduino:streamlit_ui` as run.sh checks: the child execs `<venv python> -m streamlit run --server.port <internal port> --browser.serverPort <public port> <app>/python/main.py`, since the preparation installs Streamlit in the venv. The public port is the first one the streamlit_ui brick declares in its `brick_config.yaml`, read from the installed library without importing the brick, or the smallest port of the app when that file cannot be read; when the app does not list it, Streamlit binds it directly. `start` reports `"runner":"streamlit"`.

## Preparation

The launcher installs the app dependencies before forking, only when they changed (`prepare.py`). An app has dependencies of its own when it has a non-blank `python/requirements.txt`, wheels in `python-libraries/` (`x.whl.installed`, as run.sh renames them once installed, counts as `x.whl`), a non-blank `bricks/*/requirements.txt`, or `arduino:streamlit_ui` in its `app.yaml`. Their fingerprint, the sha256 of the files' contents and of the wheel names, is recorded in `<app>/.cache/launcher/prepared.sha` after a successful preparation. A start prepares the app when it has dependencies of its own and either its venv is missing or the recorded fingerprint differs; an app without any is never prepared.

The preparation is `run.sh prepare` with `APP_DIR=<app>` and the app environment, in its own process group, its output appended to the app log. It creates the venv, installs the requirements, the wheels and Streamlit, checks the sources compile, and skips the device provisioning the serve mode already did. `start` reports `"prepared":true` and the time in `timings.prepare_ms`; a failure is a `prepare_failed` error and records nothing, so the next start tries again. A `stop` of that app or a shutdown sends SIGTERM to the preparation, the timeout SIGKILL.

The output of each app goes to `<app>/.cache/logs/app.log`, rotated at 5 MB into a single `app.log.1`. The supervisor's own lines go to its stdout, the container log.

## Fork rules

- The supervisor is single-threaded: no executors, no subprocess helpers, numeric addresses only so asyncio never resolves names in a thread. `run.sh prepare` is started with `posix_spawn` and reaped by the supervisor's own SIGCHLD handler, which also reaps the orphans the launcher inherits as PID 1: asyncio's subprocess watcher would race it for the exit status, and on macOS it waits in a thread. A fork is refused while `threading.active_count() > 1`, with the names of the threads alive. The "Ready" line also counts the OS threads on Linux, which shows native threads that `threading` cannot see.
- The fork happens between two runs of the event loop, never inside a callback, so the child never unwinds through asyncio.
- Signals are blocked across the fork and the GC is frozen in the child, so no supervisor finalizer ever runs inside the app. The child never closes the inherited event loop: on Linux the epoll instance is shared with the parent.
- Importing the library installs no signal handler; the app's `App.run()` installs its own SIGTERM handler as in its own interpreter.

## The app folder

Apps run with their own folder, mounted at its host path, as working directory, and the CLI sets `APP_HOME` to it. The library finds the folder through `arduino.app_utils.app_folder()`: `APP_HOME` when it is an existing folder, else the working directory, which is also where a developer runs an app from locally. Library code resolves it when used, never at import: the supervisor preloads the library before any app folder is known.
