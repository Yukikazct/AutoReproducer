"""Process-local compatibility for extended Windows pip target RECORD paths."""
import ntpath
import os
from pathlib import Path
import sys


def normalize_record_filesystem_path(path):
    """Keep the long-path prefix, resolving pip's migrated ../../bin records."""
    if isinstance(path, str) and path.startswith('\\\\?\\'):
        return ntpath.normpath(path)
    return path


def target_install_command(command):
    """Adapt only Windows target installs; preserve every pip CLI argument."""
    if (os.name != 'nt' or command[1:4] != ['-m', 'pip', 'install']
            or not any(isinstance(argument, str) and (argument in {'--target', '-t'} or argument.startswith('--target='))
                       for argument in command[4:])):
        return command
    # The selected interpreter loads its own pip. -I excludes the author
    # checkout and ambient PYTHONPATH, while keeping that interpreter's site.
    # -I ignores PYTHONIOENCODING; explicitly retain the owned runner's UTF-8
    # stream contract on Windows installations with a legacy system code page.
    return [command[0], '-I', '-X', 'utf8', str(Path(__file__).resolve()), *command[3:]]


def main():
    from pip._internal.operations.install import wheel
    from pip._internal.cli.main import main as pip_main

    original = wheel._record_to_fs_path

    def record_to_fs_path(record_path, lib_dir):
        # pip already checked extraction containment. A legacy .data/scripts
        # shebang rewrite causes rehash of its relative RECORD path. Win32
        # extended paths require normalization before that filesystem open.
        return normalize_record_filesystem_path(original(record_path, lib_dir))

    wheel._record_to_fs_path = record_to_fs_path
    return pip_main(sys.argv[1:])


if __name__ == '__main__':
    raise SystemExit(main())
