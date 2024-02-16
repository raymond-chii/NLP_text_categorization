import os

def read_files_in_folder(folder_path):
    lines = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines.extend(file.read().splitlines())
    return lines

folder_path = 'project 1/TC_provided/corpus1/train'
if not os.path.exists(folder_path):
    print(f"Folder '{folder_path}' does not exist.")
else:
    files_content = read_files_in_folder(folder_path)
    print(f"Read {len(files_content)} lines from folder '{folder_path}'.")
