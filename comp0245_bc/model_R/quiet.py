"""Small helpers so the scripts fail politely and keep the console readable (no walls of red text).

- quiet_native(): hides the simulator's own console spam (b3Warning, "No feet defined", ...).
  Set the environment variable ROBOENV_VERBOSE=1 to see it again when debugging.
- run_main(): runs a script's main() so that Ctrl+C or an unexpected error prints ONE clear
  line, and the full technical details go to a log file instead of the screen.
"""
import contextlib
import ctypes
import datetime
import os
import sys
import traceback

VERBOSE = os.environ.get("ROBOENV_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def _flush_c_stdio():
    """Flush C-level output buffers so hidden text cannot leak out later."""
    try:
        if os.name == "nt":
            ctypes.cdll.ucrtbase.fflush(None)
        else:
            ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


@contextlib.contextmanager
def quiet_native():
    """Temporarily send everything written to the console (Python and C code) to nowhere."""
    if VERBOSE:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        _flush_c_stdio()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved + [devnull]:
            os.close(fd)


def run_main(main, log_dir):
    """Call main(); turn crashes into one readable line plus a log file."""
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by you (Ctrl+C). Everything finished so far is saved; "
              "run the same command again to carry on.")
        sys.exit(130)
    except Exception as exc:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "error.log")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ---\n{traceback.format_exc()}\n")
        print(f"\nSomething went wrong: {type(exc).__name__}: {exc}")
        print(f"Full details were written to {log_path}")
        sys.exit(1)
