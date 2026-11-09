#!/usr/bin/env python3
"""Dedicated teacher GPU + FSDP2 attention training on the student GPUs."""
from llm_jp_logkv.fsdp_launcher import main

if __name__ == '__main__':
    main()
