import os

# Folders to clean
folders = [
    r"D:\sensi_finder\data",         # your raw labeled images
    r"D:\sensi_finder\dataset\labels"  # train + val labels
]

# The original class ID of target_bot
original_class_id = '15'

# Loop through all folders
for folder in folders:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                new_lines = []
                with open(file_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        # Keep only target_bot class, remap to 0
                        if parts[0] == original_class_id:
                            parts[0] = '0'
                            new_lines.append(" ".join(parts))
                # Overwrite file
                with open(file_path, 'w') as f:
                    f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

print("All labels remapped: target_bot is now class 0, old classes removed.")
