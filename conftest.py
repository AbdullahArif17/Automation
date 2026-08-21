"""Test configuration.

Redirect pytest's temp root to a local, writable directory. The system TEMP
directory on some Windows setups is permission-denied, which breaks the
`tmp_path` fixture. Setting tempfile.tempdir early makes tests portable.
"""
import os
import tempfile
from pathlib import Path

_LOCAL_TMP = Path(__file__).resolve().parent / ".pytest_tmp"
_LOCAL_TMP.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(_LOCAL_TMP)
os.environ["TMP"] = str(_LOCAL_TMP)
os.environ["TEMP"] = str(_LOCAL_TMP)
