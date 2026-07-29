import os
from dotenv import load_dotenv

load_dotenv()

MANAGER_BOT_TOKEN: str = os.environ["MANAGER_BOT_TOKEN"]

WAVESPEED_API_KEY: str = os.environ.get(
    "WAVESPEED_API_KEY", "wsk_live_4vuq_5tap5be0OIpNdREiYoAidSfH7bFcexfEOvO8_M"
)
WAVESPEED_BASE_URL: str = os.environ.get(
    "WAVESPEED_BASE_URL", "https://llm.wavespeed.ai/v1"
)
WAVESPEED_MODEL: str = os.environ.get("WAVESPEED_MODEL", "openai/gpt-4o-mini")

DB_PATH: str = os.environ.get("DB_PATH", "data/app.db")
