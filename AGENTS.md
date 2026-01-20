# Face System Development Guide for Agents

This document provides essential information for autonomous agents working on the Face System repository. Follow these guidelines to ensure consistency and stability.

## 1. Project Overview

This is a **Face Recognition System** built with **Python**, **PyQt5** (UI), **PyTorch** (Face embeddings), **MediaPipe** (Face detection), and **OpenVINO**. It integrates with MQTT and REST APIs for attendance tracking and access control.

- **Primary Language:** Python 3.x
- **GUI Framework:** PyQt5
- **ML Frameworks:** PyTorch, MediaPipe, OpenVINO, YOLOv10
- **Configuration:** `config.json`
- **Logging:** Custom `LOGGER` from `init.log`

## 2. Environment & Build

### Environment Setup
The project uses a virtual environment. Ensure you are using the correct Python interpreter.

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
The main entry point for the GUI application is `face_main.py`.

```bash
# Run the main face recognition system
python3 face_main.py
```

*Note: The application requires a graphical environment (X server) due to PyQt5. If running in a headless environment, expect GUI-related errors unless mocked.*

### Configuration
The system relies heavily on `config.json`.
- **DO NOT** commit changes to `config.json` containing real secrets or production IP addresses.
- If you need to add new configuration options, update `config.json` and handle default values safely in the code (e.g., using `CONFIG.get("key", default)`).

## 3. Testing & Verification

This project does not currently use a standard unit test suite (like `pytest`) for the main application logic due to its hardware-dependent nature (cameras, GPIO).

### Recognition Accuracy Test
Use `test_recognition.py` to verify face recognition accuracy. This script processes a directory of enrollment images and a directory of test images.

```bash
# Usage
python3 test_recognition.py <enroll_dir> <test_dir>

# Example
python3 test_recognition.py /path/to/enroll_images /path/to/test_images
```

### Manual Verification
When modifying UI or Camera logic:
1. Since automated UI tests are absent, verify logic by reading the relevant code paths in `CameraSystem` class in `face_main.py`.
2. Ensure no blocking calls run on the main UI thread (use `ThreadPoolExecutor` or `QThread`).

## 4. Code Style & Conventions

### Formatting
- **Indentation:** 4 spaces (Standard Python).
- **Line Length:** Soft limit of 100 characters, but longer lines are tolerated for complex logic or log messages.
- **Imports:**
  1. Standard Library (`os`, `sys`, `time`, `json`)
  2. Third-Party (`numpy`, `torch`, `cv2`, `PyQt5`)
  3. Local Modules (`init.log`, `function`, `models`)

### Naming Conventions
- **Classes:** `CamelCase` (e.g., `FaceRecognitionSystem`, `CameraSystem`)
- **Functions/Methods:** `snake_case` (e.g., `check_in_out`, `crop_face_without_forehead`)
- **Variables:** `snake_case` (e.g., `staff_name`, `img_path`)
- **Constants:** `UPPER_CASE` (e.g., `CONFIG`, `API`)

### Type Hinting
- Use type hints for new classes and complex functions.
- Refer to `GlobalState` in `face_main.py` for examples of `dataclass` usage with types.

```python
from typing import List, Dict, Any

def process_data(frame_id: int, data: Dict[str, Any]) -> bool:
    ...
```

### Logging
- **DO NOT** use `print()` for production logs.
- Use the global `LOGGER` imported from `init.log`.

```python
from init.log import LOGGER

# Good
LOGGER.info(f"User {user_id} detected with confidence {conf:.2f}")
LOGGER.error(f"Failed to connect to camera: {e}", exc_info=True)

# Avoid
print(f"User {user_id} detected")
```

### Error Handling
- Wrap hardware and network operations (Camera read, API requests) in `try...except` blocks.
- Prevent the main application from crashing due to a single camera failure.
- Log full stack traces for unexpected exceptions using `exc_info=True`.

```python
try:
    response = requests.get(api_url, timeout=5)
except Exception as e:
    LOGGER.error(f"API request failed: {e}", exc_info=True)
```

## 5. File System & Paths

- **Absolute Paths:** Always use absolute paths or resolve paths relative to `__file__`.
- **Avoid Hardcoding:** Do not hardcode paths like `/home/ubuntu/`. Use `os.path.dirname(__file__)` or configuration values.

```python
# Correct way to reference local assets
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "my_model.xml")
```

## 6. Architecture & State Management

### Global State
The application uses a `GlobalState` dataclass passed via the `system` object.
- Access state via `self.system.state`.
- Be mindful of thread safety when modifying shared state arrays (like `state.frame`, `state.same_people`).

### Threading
- **UI Thread:** Handles PyQt5 updates.
- **Camera Threads:** `CameraSystem` runs a `main_camera` loop in a separate thread.
- **IO Operations:** Use `self.system.io_pool` (ThreadPoolExecutor) for disk writes (e.g., saving images) to avoid freezing the video feed.

## 7. Critical Logic
- **Face Cropping:** `crop_face_without_forehead` in `function.py` is critical for recognition accuracy. Modifications here affect existing embeddings.
- **Attendance Logic:** `check_in_out` in `function.py` controls API calls and voice feedback. Ensure logic handles "debounce" (preventing multiple triggers) correctly.

## 8. Git & Version Control
- **Git Ignore Check (MANDATORY):** Before running `git add`, ALWAYS check the content of `.gitignore`. Never blindly add files, especially logs (`*.log`), images (`*.jpg`, `*.png`), database files (`*.db`), or virtual environment directories.
- **Handover Notes (MANDATORY):** For every new commit, you MUST append a detailed explanation of the changes (what was changed, why, and impact) to the end of `./handover_notes.md`.
  - **Do NOT delete** existing content in `handover_notes.md`; it is a critical historical record.
  - **Refinement:** If a change becomes a permanent standard (e.g., LFS check), promote the rule to this `AGENTS.md` file while keeping the history in `handover_notes.md`.
- **Commits:** Write clear, concise commit messages in **English ONLY**.
- **Secrets:** Check `git status` before adding files to ensure no temporary config files or credential files are committed.

## 9. Professional Responsibility & Edge Cases
- **No Excuses for Edge Cases:** "Edge cases" are not an excuse for system failure. To the user, a failure is a failure, regardless of the cause (blur, lighting, angle).
- **Relentless Problem Solving:** Do not dismiss difficult samples as "outliers." Use all available tools (texture analysis, frequency domain, geometric consistency) to find a solution.
- **Accountability:** If a method fails, analyze *why* mathematically and propose concrete alternatives. Do not simply state "it's impossible."
- **User-Centric Quality:** The only metric that matters is the end-user experience. If the user says it's wrong, it's wrong.

