# arduino-app-launcher

A long-lived supervisor that runs inside the container of one Arduino app (`python-apps-base`, `/run.sh serve`, its default). It imports the heavy libraries once, then starts the app as a forked child on request, so a start skips the interpreter startup and the library imports: on a VENTUNO Q a camera app logs "App started" after 1.2 s instead of 3.0 s. The container is the sandbox: one app per container, kept running after the app stops so that its next start is warm.

Run it with `arduino-app-launcher` or `python -m arduino.app_tools.app_launcher`. It opens its socket first and preloads the heavy libraries (`APP_LAUNCHER_PRELOAD`, a comma separated list of modules) only when idle, half a second after opening and one module between two turns of its loop: the start a fresh container is brought up for forks at once and the app imports what it needs itself, the next start is warm. It stops on SIGTERM or SIGINT, after stopping the app; during an import it just exits, so a stop is never lost.

With a command, the same executable is a client of the control socket that prints the JSON response (`client.py`, stdlib only, a reference for other clients):

```
arduino-app-launcher [--socket PATH] ping | status | start <app> [--env K=V ...] | stop
```

## What runs where

The daemon (arduino-app-cli) brings up the compose project of each app as it always did, its `main` service turned into a warm container, `<project>-main-1`, with docker-init as PID 1 and the supervisor as its only child; the sidecars of the project run beside it while the app runs. The app folder is bind-mounted at `/app`, the working directory, and the per-app runtime folder at `/run/arduino-app-cli`, where the control socket lives. The container publishes the public ports of the app, the ones its bricks declare and the ones of app.yaml, so Docker binds them on the host while the container runs: new ports mean a new container, and the container of an app whose ports another app's container holds stays stopped until that app starts.

The app shares its uid and the socket folder with the supervisor, so the daemon trusts neither the path nor a reply by itself: it accepts an answer only from the direct child of the container init, the supervisor, and a stop the supervisor does not confirm or complete ends with a restart of the container. The daemon starts the app through the socket, stops it the same way and leaves the container running: every app of the board has its container up from the daemon boot, so a start is a fork. A missing or stopped container is created or started through the Docker API, the cold path, before the start request.

## Processes

- **docker-init** (PID 1): reaps the orphans and forwards the container stop to the supervisor.
- **Supervisor** (`supervisor.py`): one asyncio loop in one thread. Owns the control socket and the one app process. It listens on no TCP port.
- **App child** (`child.py`, stdlib only): forked from the supervisor, it becomes the app the way `python main.py` would run it.
- **Preparation**: `run.sh prepare`, spawned by the supervisor before a start when the app dependencies changed.

## Control socket

A Unix socket at `APP_LAUNCHER_SOCKET`, mode 0660. One JSON object per line, one request per connection, then the response and the connection closes. Lifecycle requests (`start`, `stop`) run one at a time; `status` and `ping` answer immediately.

| Request | Response |
|---|---|
| `{"cmd":"start","app":"/app","env":{"K":"V"}}` | `{"ok":true,"pid":N,"started_at":"<iso>","warm":true,"runner":"python","prepared":false,"timings":{"stop_ms":..,"prepare_ms":..,"fork_ms":..}}` |
| `{"cmd":"stop"}` | `{"ok":true,"stopped":[{"app":..,"pid":..,"exit_code":..,"signal":..}]}` |
| `{"cmd":"status"}` | `{"ok":true,"child":{"app","pid","runner","warm","started_at","stopping"} or null,"preload":{"modules":[..],"loaded":[..],"failed":[..]},"version":..,"uptime_s":..}` |
| `{"cmd":"ping"}` | `{"ok":true}` |

`start` takes the absolute app folder and an optional `env`, an object of strings: the app environment. A start with an app running stops it first, a reload, and reports the time in `timings.stop_ms`; then it prepares the app when needed and forks it. `stop` stops the app, `stopped` holds zero or one entry; an `app` in the request is ignored. Stopping when nothing runs is not an error.

Errors are `{"ok":false,"error":{"code":..,"message":..}}` with `code` one of `bad_request` (malformed request, relative app path, `env` not an object of strings), `not_found` (no app folder or no `python/main.py`), `busy` (the launcher is shutting down), `prepare_failed` (`run.sh prepare` failed, was stopped or timed out, with the last lines of its output), `internal` (fork refused or failed, unexpected errors).

## Stopping and what the app leaves behind

A stop sends SIGTERM to the app's process group, waits `APP_LAUNCHER_STOP_TIMEOUT_S`, then SIGKILLs it. Once the app is gone, every other process of the container is killed but PID 1 and the supervisor, helpers in sessions of their own included, listed again from `/proc` for the ones forked meanwhile; zombies are left to docker-init. The same sweep follows the end of a preparation and an app that exits on its own. The sweep runs only when the supervisor tops a pid namespace of its own, as PID 1 or the child of PID 1: elsewhere, a development machine or a test, the other processes are not the app's and only the process group is killed.

## Environment

The child runs with the container environment updated by the `env` of the start request, then the additions run.sh made in the classic stack: `PYTHONUNBUFFERED=1`, `<app>/bricks` on `PYTHONPATH` when it exists, and `VIRTUAL_ENV` with its `bin` first on `PATH` when the app runs in its venv. The preparation gets the same environment, before those additions, plus `APP_DIR=<app>`.

| Variable | Default | |
|---|---|---|
| `APP_LAUNCHER_PRELOAD` | `arduino.app_utils,arduino.app_peripherals.camera,arduino.app_peripherals.microphone,arduino.app_peripherals.speaker,arduino.app_bricks.web_ui` | Modules imported at startup, one by one; a failing one is logged and reported by `status` |
| `APP_LAUNCHER_SOCKET` | `/run/arduino-app-cli/app-launcher.sock` | Control socket; a stale one is replaced |
| `APP_LAUNCHER_STOP_TIMEOUT_S` | `5` | Grace period between SIGTERM and SIGKILL |
| `APP_LAUNCHER_RUN_SH` | `/run.sh` | The entrypoint whose `prepare` mode installs the app dependencies |
| `APP_LAUNCHER_PREPARE_TIMEOUT_S` | `600` | How long a preparation may take before it is killed; the client waits this long plus a minute for a `start` |

## Logs

The app inherits the container stdout and stderr: what it prints is read with `docker logs`, as in the classic stack. The supervisor writes its own lines to stderr, prefixed with `app-launcher:`, warnings only unless `APP_LAUNCHER_LOG_LEVEL` asks for `info` or `debug`, and relays the preparation output to stdout.

## The app child

Before any app code runs, the child:

1. becomes a session and process group leader;
2. points every inherited fd above 2 at `/dev/null`, keeping the container stdin, stdout and stderr;
3. gets back the signal dispositions the supervisor had after the preload, which are those of a fresh `python main.py`, and reseeds NumPy's global generator, which the fork copied from the warm interpreter;
4. moves to the app folder, takes the environment above and sets `sys.argv` and `sys.path` as a fresh interpreter would;
5. runs `python/main.py` as a permanent `__main__` module: a fresh module with the dunders of a script, not `runpy`, so the app's threads and atexit handlers keep seeing it. Uncaught exceptions go through `sys.excepthook` without the launcher frames, with exit code 1; `SystemExit` is honoured and the interpreter exits normally.

Two kinds of apps are not run warm: the child execs instead, with the same environment, cwd and process group, and `start` reports `"warm":false`.

- An app with dependencies of its own, as the preparation defines them, or whose venv `<app>/.cache/.venv` holds a `*.dist-info` or a `*.pth` other than the `_virtualenv.pth` of an empty `uv venv`. The child execs the venv interpreter on `main.py`.
- A Streamlit app, whose `app.yaml` mentions `arduino:streamlit_ui` as run.sh checks: the child execs `<venv python> -m streamlit run --server.port 7000 <app>/python/main.py`, the port run.sh always gave Streamlit, since the preparation installs Streamlit in the venv. `start` reports `"runner":"streamlit"`.

## Preparation

The launcher installs the app dependencies before forking, only when they changed (`prepare.py`). An app has dependencies of its own when it has a non-blank `python/requirements.txt`, wheels in `python-libraries/` (`x.whl.installed`, as run.sh renames them once installed, counts as `x.whl`), a non-blank `bricks/*/requirements.txt`, or `arduino:streamlit_ui` in its `app.yaml`. Their fingerprint, the sha256 of the files' contents and of the wheel names, is recorded in `<app>/.cache/prepared.sha` after a successful preparation. A start prepares the app when it has dependencies of its own and either its venv is missing or the recorded fingerprint differs; an app without any is never prepared.

The preparation is `run.sh prepare` in its own process group. It creates the venv, installs the requirements, the wheels and Streamlit, and checks the sources compile. Its output goes through a pipe the supervisor relays to stdout, keeping the tail for the `prepare_failed` message. `start` reports `"prepared":true` and the time in `timings.prepare_ms`; a failure records nothing, so the next start tries again. A `stop` or a shutdown sends SIGTERM to the preparation, the timeout SIGKILL.

## Fork rules

- The supervisor is single-threaded: no executors, no subprocess helpers, numeric addresses only so asyncio never resolves names in a thread. `run.sh prepare` is started with `posix_spawn` and reaped by the supervisor's own SIGCHLD handler: asyncio's subprocess watcher would race it for the exit status, and on macOS it waits in a thread. A fork is refused while `threading.active_count() > 1`, with the names of the threads alive. The "Ready" line also counts the OS threads on Linux, which shows native threads that `threading` cannot see.
- The fork happens between two runs of the event loop, never inside a callback, so the child never unwinds through asyncio.
- Signals are blocked across the fork and the GC is frozen in the child, so no supervisor finalizer ever runs inside the app. The child never closes the inherited event loop: on Linux the epoll instance is shared with the parent.
- Importing the library installs no signal handler; the app's `App.run()` installs its own SIGTERM handler as in its own interpreter.

## The app folder

The library finds the app folder through `arduino.app_utils.app_folder()`: the nearest folder holding an `app.yaml` above the running script, else above the working directory, else the working directory itself, which is also where a developer runs an app from locally. In the container that is `/app`. Library code resolves it when used, never at import: the supervisor preloads the library before any app runs.
