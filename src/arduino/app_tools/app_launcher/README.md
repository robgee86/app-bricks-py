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

A request from a process the launcher itself spawned, an app or anything an app left behind, is refused with `forbidden`: the peer's pid, which the kernel attaches to the connection, descends from the launcher. The daemon and the CLI connect from outside the container's pid namespace, where that pid reads 0, and a `docker exec` client, the health check, descends from outside too. On macOS, where no peer pid exists, nothing is refused.

| Request | Response |
|---|---|
| `{"cmd":"start","app":"<abs folder>","env":{...},"mode":"replace","ports":[7000]}` | `{"ok":true,"pid":N,"started_at":"<iso>","warm":true,"runner":"python","confined":true,"prepared":false,"timings":{"stop_ms":..,"prepare_ms":..,"fork_ms":..}}` |
| `{"cmd":"stop"}` or `{"cmd":"stop","app":"<abs folder>"}` | `{"ok":true,"stopped":[{"app":..,"pid":..,"exit_code":..,"signal":..}]}` |
| `{"cmd":"status"}` | `{"ok":true,"confinement":{"required":true,"landlock_abi":N},"children":[{"app","pid","state","started_at","runner","warm"}],"preload":{"modules":[..],"failed":[..]},"version":..,"uptime_s":..}` |
| `{"cmd":"ping"}` | `{"ok":true}` |

`start` with `replace` stops every running app first, prepares the new one when needed, then forks it; its `ports` are the TCP ports the app may bind, the ones it declares, none when absent, see [Confinement](#confinement). `mode` `add` is refused with `unsupported`: one app at a time for now. Stopping an app that is not running is not an error. A stop sends SIGTERM, waits the grace period, then SIGKILLs the app's process group; whatever the app leaves in its group is killed once it exits, as in a container whose main process ends.

Errors are `{"ok":false,"error":{"code":..,"message":..}}` with `code` one of `bad_request` (malformed request, relative app path), `not_found` (no app folder or no `python/main.py`), `unsupported` (`mode` `add`, or confinement required on a kernel without Landlock), `forbidden` (a request from a process of an app), `busy` (the launcher is shutting down), `prepare_failed` (`run.sh prepare` failed, was stopped or timed out, with the last lines of its output), `internal` (fork refused or failed, unexpected errors).

## Environment

| Variable | Default | |
|---|---|---|
| `APP_LAUNCHER_PRELOAD` | `arduino.app_utils,arduino.app_peripherals.camera,arduino.app_peripherals.microphone,arduino.app_peripherals.speaker,arduino.app_bricks.web_ui` | Modules imported at startup, one by one; a failing one is logged and reported by `status` |
| `APP_LAUNCHER_SOCKET` | `/run/arduino-app-cli/app-launcher.sock` | Control socket; a stale one is replaced |
| `APP_LAUNCHER_STOP_TIMEOUT_S` | `5` | Grace period between SIGTERM and SIGKILL, the CLI's container stop timeout |
| `APP_LAUNCHER_RUN_SH` | `/run.sh` | The entrypoint whose `prepare` mode installs the app dependencies |
| `APP_LAUNCHER_PREPARE_TIMEOUT_S` | `600` | How long a preparation may take before it is killed; the client waits this long plus a minute for a `start` |
| `APP_LAUNCHER_CONFINEMENT` | `required` | `required` confines every app and preparation with Landlock and starts nothing on a kernel without it; `off` is for development machines, macOS included |
| `APP_LAUNCHER_APP_ALIAS` | `/app` | The alias of the running app, a link to the link the supervisor points at the app folder, see [The app folder](#the-app-folder) |

## Ports

The launcher publishes no port: the CLI's ingress does, forwarding each public port of the app to the same port in the launcher's container, where the app binds it. The `ports` of the `start` request are those ports, the ones the app's bricks declare, and the only TCP ports the confined app may bind.

## The app child

Before any app code runs, the child:

1. becomes a session and process group leader;
2. points every inherited fd above 2 at `/dev/null` and its stdout and stderr at the log pipe;
3. gets back the signal dispositions the supervisor had after the preload, which are those of a fresh `python main.py`;
4. gets the supervisor's environment with the request's `env` on top, `HOME` set to the app's own home, see [The app home](#the-app-home), then `PYTHONUNBUFFERED=1`, and `<app>/bricks` on `PYTHONPATH` when it exists;
5. moves to the app folder, confines itself to it, see [Confinement](#confinement), and sets `sys.argv` and `sys.path` as a fresh interpreter would;
6. runs `python/main.py` as a permanent `__main__` module: a fresh module with the dunders of a script, not `runpy`, so the app's threads and atexit handlers keep seeing it. Uncaught exceptions go through `sys.excepthook` without the launcher frames, with exit code 1; `SystemExit` is honoured and the interpreter exits normally.

Two kinds of apps are not run warm: the child execs instead, with the same environment, cwd and process group, and `start` reports `"warm":false`.

- An app with dependencies of its own, as the preparation defines them, or whose venv `<app>/.cache/.venv` holds a `*.dist-info` or a `*.pth` other than the `_virtualenv.pth` of an empty `uv venv`. The child execs the venv interpreter on `main.py`, with `VIRTUAL_ENV` set and its `bin` first on `PATH`.
- A Streamlit app, whose `app.yaml` mentions `arduino:streamlit_ui` as run.sh checks: the child execs `<venv python> -m streamlit run --server.port 7000 <app>/python/main.py`, the port run.sh always gave Streamlit, since the preparation installs Streamlit in the venv. `start` reports `"runner":"streamlit"`.

## Confinement

Every app runs confined to its own folder with Landlock, the kernel's unprivileged sandbox, applied by the child itself right before the app code runs (`confinement.py`) and by the preparation before `run.sh prepare`. No capability or profile is involved: the process restricts itself, a later ruleset can only tighten it, and the restriction follows exec and fork. It is an allow-list:

- read and write: the app folder, its home included, `/dev` with its ioctls, `/tmp` and `XDG_RUNTIME_DIR`;
- read and execute: `/usr`, `/lib`, `/bin`, `/sbin`, `/etc`, `/opt`, `/sys`, `/proc`, `/run.sh`, the image home `/home/app` with its fonts, and `MODELS_PATH` when set;
- bind: the TCP ports of the `start` request, plus an ephemeral port of the kernel's choice, which libraries use for helper servers of their own;
- nothing else: the other apps' folders and the apps root cannot be read, listed or written, the environment and the memory of other processes cannot be read through `/proc`, and on kernels with Landlock ABI 6 or later no signal reaches a process outside the app and no abstract Unix socket of another process accepts it.

`/tmp` and `/dev/shm` stay shared among the apps of the container, the latter holding the semaphores and shared memory of `multiprocessing`: what one app writes there, another can read. Landlock does not mediate connecting to a Unix socket by its path: a server reached that way checks its clients itself, as the launcher does with the peer pid and the TPS scanner with a token in the app folder.

`status` reports the ABI in use. With `APP_LAUNCHER_CONFINEMENT=required`, the default, a kernel without Landlock starts no app: `start` is refused with `unsupported`.

## Preparation

The launcher installs the app dependencies before forking, only when they changed (`prepare.py`). An app has dependencies of its own when it has a non-blank `python/requirements.txt`, wheels in `python-libraries/` (`x.whl.installed`, as run.sh renames them once installed, counts as `x.whl`), a non-blank `bricks/*/requirements.txt`, or `arduino:streamlit_ui` in its `app.yaml`. Their fingerprint, the sha256 of the files' contents and of the wheel names, is recorded in `<app>/.cache/launcher/prepared.sha` after a successful preparation. A start prepares the app when it has dependencies of its own and either its venv is missing or the recorded fingerprint differs; an app without any is never prepared.

The preparation is `run.sh prepare` with `APP_DIR=<app>` and the app environment, its home included, in its own process group, its output appended to the app log, through the confinement module run as an isolated interpreter (`python -I`), which imports from the image only. The installs follow run.sh and the image environment, no `uv.toml` or `pyproject.toml` of the app or the home steers them (`UV_NO_CONFIG=1`), and no user site-packages is ever on the path of the launcher, the apps or their venvs (`PYTHONNOUSERSITE=1`, set by run.sh). It creates the venv, installs the requirements, the wheels and Streamlit, checks the sources compile, and skips the device provisioning the serve mode already did. `start` reports `"prepared":true` and the time in `timings.prepare_ms`; a failure is a `prepare_failed` error and records nothing, so the next start tries again. A `stop` of that app or a shutdown sends SIGTERM to the preparation, the timeout SIGKILL.

The output of each app goes to `<app>/.cache/logs/app.log`, rotated at 5 MB into a single `app.log.1`; every start begins a new log, so it holds the current run only. The supervisor's own lines go to its stdout, the container log.

## Fork rules

- The supervisor is single-threaded: no executors, no subprocess helpers, numeric addresses only so asyncio never resolves names in a thread. `run.sh prepare` is started with `posix_spawn` and reaped by the supervisor's own SIGCHLD handler, which also reaps the orphans the launcher inherits as PID 1: asyncio's subprocess watcher would race it for the exit status, and on macOS it waits in a thread. A fork is refused while `threading.active_count() > 1`, with the names of the threads alive. The "Ready" line also counts the OS threads on Linux, which shows native threads that `threading` cannot see.
- The fork happens between two runs of the event loop, never inside a callback, so the child never unwinds through asyncio.
- Signals are blocked across the fork and the GC is frozen in the child, so no supervisor finalizer ever runs inside the app. The child never closes the inherited event loop: on Linux the epoll instance is shared with the parent.
- Importing the library installs no signal handler; the app's `App.run()` installs its own SIGTERM handler as in its own interpreter.

## The app folder

Apps run with their own folder, mounted at its host path, as working directory. The library finds the folder through `arduino.app_utils.app_folder()`: the nearest folder holding an `app.yaml` above the running script, else above the working directory, else the working directory itself, which is also where a developer runs an app from locally. No environment variable is involved. Library code resolves it when used, never at import: the supervisor preloads the library before any app folder is known.

## The app home

One container runs every app, so a home shared among them would carry what one app writes, a uv configuration or a Python package, into the next app and its preparation. Each app runs with `HOME=<app>/.cache/home` instead, a folder of its own that the supervisor creates before each start and fills as a new user's home: the folders of `/etc/skel` are created in it and its files linked, the Streamlit theme and the ALSA wrappers the image puts there, so an image upgrade reaches every app while what the app adds, caches and dotfiles, stays. The image home `/home/app` is read-only for apps: it holds the fonts the library reads and the link behind `/app`.

`/app` keeps naming the running app, for the apps and tools written against the folder the classic stack mounted there. In the image `/app` is a link to a second link, which the supervisor points at the app folder before each start: a restart of the same app finds it in place, a switch repoints it, and it goes away when no app runs any more, after a stop, when the last app exits on its own, and at startup. The supervisor finds the second link by reading `/app` (`APP_LAUNCHER_APP_ALIAS` names another alias outside the image); where the alias is not a link, nothing is managed. It is an alias only: the library never reads it, and files written through it land in the real folder.
