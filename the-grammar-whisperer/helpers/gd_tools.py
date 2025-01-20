import os
from datetime import datetime, timedelta
from google.colab import drive
from importlib import import_module


def remount_gdrive(modules=[]):

    drive.mount("/content/drive", force_remount=True)

    for module in modules:
        import_module(module)


def ls(folder, sort_order="newest", color=True):
    def format_timedelta(td):
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days > 0:
            return f"{days}d:{hours:02}h:{minutes:02}m:{seconds:02}s ago"
        elif hours > 0:
            return f"{hours:02}h:{minutes:02}m:{seconds:02}s ago"
        elif minutes > 0:
            return f"{minutes:02}m:{seconds:02}s ago"
        else:
            return f"{seconds}s ago"

    def color_text(text, delta):
        if delta < timedelta(seconds=30):
            return f"\033[92m{text}\033[0m"  # Green
        elif delta < timedelta(seconds=60):
            return f"\033[94m{text}\033[0m"  # Yellow
        else:
            return f"\033[91m{text}\033[0m"  # Red

    now = datetime.now()
    max_length = 0
    file_mod_times = []
    exclude = [".pyc", ".ipynb"]
    for root, dirs, files in os.walk(folder):
        for name in files:
            if any(substring in name for substring in exclude):
                continue
            filepath = os.path.join(root, name)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            delta = now - mod_time
            formatted_time = format_timedelta(delta)
            file_mod_times.append((delta, formatted_time, name))
            max_length = max(max_length, len(formatted_time))

    if sort_order == "newest":
        file_mod_times.sort(key=lambda x: x[0])
    elif sort_order == "oldest":
        file_mod_times.sort(key=lambda x: x[0], reverse=True)

    for delta, formatted_time, name in file_mod_times:
        if color:
            name = color_text(name, delta)
        print(f"{formatted_time.rjust(max_length)} {name}")
