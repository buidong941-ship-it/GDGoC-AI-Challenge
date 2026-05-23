import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT_DIR = Path(__file__).resolve().parent.parent

# Vietnam Timezone (UTC+7)
VIETNAM_TZ = timezone(timedelta(hours=7))

def get_vietnam_now():
    """Get the current time in Vietnam timezone."""
    return datetime.now(VIETNAM_TZ)

def load_env():
    """Load environment variables from .env file."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value

# Automatically load when config is imported
load_env()
