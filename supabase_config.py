import os

from dotenv import load_dotenv
from supabase import create_client


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "rostros").strip()
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "perfiles_sface").strip()

if not SUPABASE_URL:
    raise RuntimeError("Falta SUPABASE_URL en el archivo .env")

if not SUPABASE_KEY:
    raise RuntimeError("Falta SUPABASE_KEY en el archivo .env")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)
