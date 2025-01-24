import os
import glob
import re
import pandas as pd


def save_checkpoint(base_filename, data, N=3):
    """Save a checkpoint file while maintaining only the last N versions."""
    # Get existing checkpoints
    name, ext = os.path.splitext(base_filename)
    pattern = f"{name}_checkpoint[0-9]*{ext}"
    existing_files = glob.glob(pattern)

    # Extract checkpoint numbers
    numbers = []
    for file in existing_files:
        match = re.search(rf"{name}_checkpoint(\d+){ext}", file)
        if match:
            numbers.append(int(match.group(1)))

    # Determine next checkpoint number
    next_num = 1 if not numbers else max(numbers) + 1

    # Save new checkpoint
    checkpoint_name = f"{name}_checkpoint{next_num}{ext}"
    data.to_csv(checkpoint_name, index=False)

    # Remove old checkpoints if more than N
    if len(existing_files) >= N:
        checkpoint_files = [(f, int(re.search(rf"{name}_checkpoint(\d+){ext}", f).group(1))) for f in existing_files]
        checkpoint_files.sort(key=lambda x: x[1])

        # Remove oldest files until only N-1 remain (plus the new one we just added)
        while len(checkpoint_files) >= N:
            os.remove(checkpoint_files[0][0])
            checkpoint_files.pop(0)

    return checkpoint_name


def load_checkpoint(base_filename, sep=","):
    """Load the latest checkpoint file."""
    name, ext = os.path.splitext(base_filename)
    pattern = f"{name}_checkpoint[0-9]*{ext}"
    existing_files = glob.glob(pattern)

    if not existing_files:
        if os.path.exists(base_filename):
            return pd.read_csv(base_filename, sep=sep), base_filename
        else:
            return None, None

    # Find the highest checkpoint number
    checkpoint_files = [(f, int(re.search(rf"{name}_checkpoint(\d+){ext}", f).group(1))) for f in existing_files]
    latest_file = max(checkpoint_files, key=lambda x: x[1])[0]

    # Your existing load function here, e.g.:
    return pd.read_csv(latest_file, sep=sep), latest_file
