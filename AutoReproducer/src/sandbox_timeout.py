"""Standalone supervisor copied into Docker; requires only the standard library."""
import json
import subprocess
import sys

MARKER = "AUTOREPRO_PHASE_RESULT:"


def main(argv=None):
    phase, seconds, *command = list(sys.argv[1:] if argv is None else argv)
    limit = float(seconds)
    try:
        result = subprocess.run(command, timeout=limit)
        code = result.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        code = 124
        timed_out = True
    if code:
        # Preserve normal stdout/stderr. The host removes this structured marker
        # and explains whether installation or actual program execution failed.
        print(MARKER + json.dumps({"phase": phase, "timeout": timed_out,
                                  "seconds": limit, "returncode": code}),
              file=sys.stderr, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
