"""
This program demonstrates basic file operations in Python.
It creates a file, writes content to it, reads the content,
and then closes the file properly.
"""

import os  # Import the os module for interacting with the operating system


def create_file(filename, content):
    """
    Creates a new file with the specified filename and writes the given content to it.

    Args:
        filename (str): The name of the file to create.
        content (str): The content to write to the file.
    """
    try:
        with open(filename, 'w') as f:  # Open the file in write mode ('w')
            f.write(content)  # Write the content to the file
        print(f"File '{filename}' created successfully.")
    except Exception as e:
        print(f"Error creating file: {e}")


def read_file(filename):
    """
    Reads the content of the specified file and returns it as a string.

    Args:
        filename (str): The name of the file to read.

    Returns:
        str: The content of the file as a string, or None if the file does not exist
             or an error occurred.
    """
    try:
        with open(filename, 'r') as f:  # Open the file in read mode ('r')
            content = f.read()  # Read the entire content of the file
            return content
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def main():
    """
    Main function to demonstrate the file operations.
    """
    filename = "test.txt"
    content = "This is some sample content for the file.\nIt can have multiple lines."

    # Create the file
    create_file(filename, content)

    # Read the file
    read_content = read_file(filename)

    if read_content is not None:
        print("\nFile Content:")
        print(read_content)

    # Clean up (optional - remove the file after testing)