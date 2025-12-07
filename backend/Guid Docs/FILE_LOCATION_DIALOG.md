# File Location Dialog Feature

## Overview

When creating files in Agent Mode, a dialog automatically pops up to let you specify:
- **Filename**: The name of the file (with extension)
- **Directory/Path**: Where the file should be saved

This gives you full control over file organization and location.

## How It Works

### Automatic Detection

The system automatically detects file creation tasks when you:
- Use Agent Mode
- Include keywords like "create", "write", "make" with "file", "python", "code", "essay", or "text"

### Dialog Features

1. **Smart Filename Detection**: Automatically extracts suggested filename from your message
2. **Directory Selection**: Choose where to save the file
3. **Path Preview**: See the full path before confirming
4. **Validation**: Ensures filename is provided before creating

## Usage Examples

### Example 1: Python File Creation

**User Input:**
```
Create a python file called calculator.py and write code about a calculator
```

**What Happens:**
1. Dialog appears automatically
2. Filename pre-filled: `calculator.py`
3. Directory: `agent_outputs` (default)
4. Full path shown: `agent_outputs/calculator.py`
5. You can modify the path before confirming

### Example 2: Custom Directory

**User Input:**
```
Write python code for a web scraper
```

**Dialog Options:**
- Filename: `code.py` (auto-suggested)
- Directory: You can change to `projects/web_scraper` or any path
- Full path: `projects/web_scraper/code.py`

### Example 3: Absolute Path

**User Input:**
```
Create a file called notes.txt
```

**Dialog Options:**
- Filename: `notes.txt`
- Directory: Can use absolute path like `/home/user/documents`
- Full path: `/home/user/documents/notes.txt`

## Dialog Interface

The dialog includes:

1. **Filename Input**
   - Text field for filename
   - Auto-filled with detected filename
   - Must include file extension (.py, .txt, .md, etc.)
   - Placeholder: "e.g., calculator.py, notes.txt"

2. **Directory/Path Input**
   - Text field for directory path
   - Default: `agent_outputs`
   - Can be relative or absolute
   - Placeholder: "e.g., agent_outputs, projects/myapp, /home/user/docs"

3. **Path Preview**
   - Shows full path: `{directory}/{filename}`
   - Updates in real-time as you type
   - Helps verify the location

4. **Action Buttons**
   - **Cancel**: Closes dialog without creating file
   - **Create File**: Confirms and creates the file

## Supported File Types

The dialog works with all file types:
- ✅ Python files (`.py`)
- ✅ Text files (`.txt`, `.md`)
- ✅ Code files (`.js`, `.html`, `.css`, `.json`)
- ✅ Any file extension you specify

## Path Formats

### Relative Paths
- `agent_outputs` → Saves to `backend/agent_outputs/`
- `projects/myapp` → Saves to `backend/projects/myapp/`
- `docs/notes` → Saves to `backend/docs/notes/`

### Absolute Paths
- `/home/user/documents` → Saves to absolute path
- `/tmp/files` → Saves to system temp directory
- `~/projects` → Saves to user home directory (if supported)

## Smart Features

### Automatic Filename Detection

The system tries to extract filename from your message:

**Examples:**
- `"Create calculator.py"` → Detects `calculator.py`
- `"Write code in file scraper.py"` → Detects `scraper.py`
- `"Create a file called notes.txt"` → Detects `notes.txt`

### Default Behavior

If no filename is detected:
- Python/code requests → `code_{timestamp}.py`
- Text/essay requests → `essay_{timestamp}.txt`

## Integration with Task Chaining

The dialog works seamlessly with task chaining:

**Example:**
```
Create calculator.py, then send email to team@company.com
```

**What Happens:**
1. Dialog appears for file creation
2. After confirming, file is created
3. Email task executes next

## Backend Processing

The backend receives the file path in the format:
```
{original_message} (save to: {directory}/{filename})
```

The task parser:
1. Extracts the path from `(save to: ...)`
2. Uses it as the file location
3. Removes the path instruction from message for cleaner parsing

## File Location

Files are saved relative to the backend directory by default:
- Default: `backend/agent_outputs/`
- Custom: `backend/{your_path}/`

## Tips

1. **Always include extension**: Make sure filename includes `.py`, `.txt`, etc.
2. **Use descriptive paths**: Organize files with meaningful directory structures
3. **Preview before confirming**: Check the full path preview
4. **Cancel if needed**: You can cancel and modify your request

## Example Workflow

1. **User types**: "Create a python file called calculator.py"
2. **Dialog appears** with:
   - Filename: `calculator.py` (pre-filled)
   - Directory: `agent_outputs` (default)
   - Preview: `agent_outputs/calculator.py`
3. **User can modify**:
   - Change to `projects/calculator.py`
   - Or keep default
4. **User clicks "Create File"**
5. **File is created** at specified location
6. **Agent generates code** and saves it

## Summary

| Feature | Status |
|---------|--------|
| Automatic Detection | ✅ Yes |
| Filename Input | ✅ Yes |
| Directory Selection | ✅ Yes |
| Path Preview | ✅ Yes |
| Smart Suggestions | ✅ Yes |
| Task Chaining | ✅ Yes |

The file location dialog gives you full control over where files are saved! 📁✨



