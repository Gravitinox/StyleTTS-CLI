import os
import sys

def append_to_lines(filename):
    # Open the file for reading and writing
    with open(filename, 'r+') as file:
        # Read all lines from the file
        lines = file.readlines()
        
        # Move the file pointer to the beginning of the file
        file.seek(0)
        
        # Append '|0' to each line and write it back to the file
        for line in lines:
            file.write(line.strip() + '|0\n')
        
        # Truncate the file to the current position (in case the new content is shorter)
        file.truncate()

def main(directory):
    files_to_process = ['train.txt', 'validation.txt']
    
    for file_name in files_to_process:
        file_path = os.path.join("training", directory, file_name)
        append_to_lines(file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    main(directory)