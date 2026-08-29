#!/usr/bin/env python3
"""PyInstaller entry for Vocalis — avoids relative-import failure of __main__.py."""
from vocalis.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
