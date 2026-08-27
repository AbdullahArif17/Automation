#!/usr/bin/env python3
"""Validate YouTube cookies file format and authentication.

Usage: python validate_cookies.py <cookies_path> [test_url]
Exits 0 on success, 1 on validation failure.
"""
import sys
from pathlib import Path

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.clipper.storage_poller import validate_cookies_file


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: validate_cookies.py <cookies_path> [test_url]", file=sys.stderr)
        return 2

    cookies_path = Path(sys.argv[1])
    test_url = sys.argv[2] if len(sys.argv) > 2 else "https://www.youtube.com/"

    try:
        count, auth_ok = validate_cookies_file(cookies_path, test_url)
        print(f"COOKIE_VALIDATION: {count} cookies, auth_ok={auth_ok}")
        if not auth_ok:
            print("ERROR: Cookies failed authentication check", file=sys.stderr)
            return 1
    except ValueError as e:
        print(f"ERROR: Cookie format validation failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())