import os
import sys
import tempfile

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep the API's session DB out of the repo during tests.
os.environ.setdefault("SESSION_DB", os.path.join(tempfile.gettempdir(), "legal-rag-test-sessions.db"))