 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.6
Date: 10-02-2026
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import os


class FileDirectoryWorker():
    def __init__(self):
        super().__init__()

    def log_stats(self, project_name: str, results = None):
        file = open(f'{project_name}_log.txt', 'a')
        file.write(results + "\n")
        file.close()

    def create_dir(self, project_name: str):
        os.makedirs(project_name, exist_ok = True)

    def create_dir_log(self, project_name: str):
        os.makedirs(project_name, exist_ok = True)        
