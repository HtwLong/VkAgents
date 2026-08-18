import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import uvicorn

load_dotenv()

SRC_DIR = Path(__file__).resolve().parent

sys.path.append(str(SRC_DIR))

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(SRC_DIR)],
    )
