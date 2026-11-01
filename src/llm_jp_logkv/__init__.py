"""Tools for investigating pretrained Llama to LogKV conversion.

Keep this module free of heavyweight imports: configure storage before loading
Hugging Face libraries, which read cache environment variables at import time.
"""
