import os
import sys

def replace_text_in_file(file_path):
    try:
        # Attempt to open the file with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        # If utf-8 fails, fall back to latin-1 encoding
        with open(file_path, 'r', encoding='latin-1') as file:
            lines = file.readlines()
    
    # Replace ".|0" at the end of each line with "ABC"
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in lines:
                if line.endswith(" |0\n"):
                    line = line[:-4] + "|0\n"
                elif line.endswith(" |0"):
                    line = line[:-3] + "|0"
                file.write(line)
        
        print(f"Processed file: {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to {file_path}: {e}")

def main(directory):
    files_to_process = ['train_phoneme.txt', 'validation_phoneme.txt']
    
    for file_name in files_to_process:
        file_path = os.path.join("training", directory, file_name)
        replace_text_in_file(file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    main(directory)
