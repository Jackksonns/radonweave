# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""Configuration Management module
"""
import os
import json

CONFIG = {}

CONFIG_FILE = os.path.expanduser("~/.dival/config.json") #use dival as an example

def init_config():
    
    global CONFIG
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                CONFIG.update(json.load(f))
        except json.JSONDecodeError:
            pass

def set_config(key, value):
    
    keys = key.split('/')
    d = CONFIG
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(CONFIG, f, indent=2)

init_config()