import os
from nltk.tokenize import word_tokenize


def read_file(folder):
    files_content = {}
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as file:
            files_content[filename] = file.read()
    return files_content

def tokenize_file(file_content):
    tokens = {}
    for filename, file_content in file_content.items():
        tokens[filename] = word_tokenize(file_content)
    return tokens


train_folder = 'project 1/TC_provided/corpus1/train'

if not os.path.exists(train_folder):
    print(f"Folder '{train_folder}' does not exist.")
else:
    files_content = read_file(train_folder)
    tokens_content = tokenize_file(files_content)
    print(tokens_content)
    print(f"Read {len(files_content)} files from folder '{train_folder}'.")




