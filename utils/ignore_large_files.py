import os

max_size_mb = 10  # Set the maximum allowed file size in MB

# Walk through the repo and find files larger than the specified size
large_files = []
for root, _, files in os.walk('.'):
    for file in files:
        file_path = os.path.join(root, file)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            large_files.append(file_path)

# Append the large files to the .gitignore file
with open('.gitignore', 'a') as gitignore:
    for large_file in large_files:
        gitignore.write(f'{large_file}\n')
