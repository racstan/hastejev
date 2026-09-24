#!/usr/bin/env python3
"""Legacy entrypoint — delegates to export_hf_models.py (honest cards + true quant keys)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_hf_models import main

if __name__ == "__main__":
    main()
