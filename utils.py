# -*- coding: utf-8 -*-

from pathlib import Path

def get_unique_file(file):
    return list(Path.cwd().rglob(file))[0]

def get_unique_directory(directory):
    directories = list(Path.cwd().rglob(directory))
    if directories[0].is_dir(): return list(Path.cwd().rglob(directory))[0]