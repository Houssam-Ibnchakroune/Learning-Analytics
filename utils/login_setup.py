import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(level_console=logging.INFO, level_file=logging.DEBUG):
    os.makedirs("logs", exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setLevel(level_console)
    console.setFormatter(fmt)

    fileh = RotatingFileHandler("logs/app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fileh.setLevel(level_file)
    fileh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:      
        root.addHandler(console)
        root.addHandler(fileh)
    root.debug("Logging is set up.")  