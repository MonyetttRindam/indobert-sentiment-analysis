"""Entry point untuk Streamlit Community Cloud.

Streamlit Cloud secara default menjalankan `streamlit_app.py` di root repo.
Aplikasi sebenarnya ada di `app/tomlembong.py`, jadi file ini hanya
menjalankan script tersebut apa adanya (dieksekusi ulang tiap rerun,
sama seperti kalau ia jadi main file).
"""

import runpy
from pathlib import Path

APP_PATH = Path(__file__).parent / "app" / "tomlembong.py"

runpy.run_path(str(APP_PATH), run_name="__main__")
