# Blum bot

Added Christmas event support!

Performant clicker for Blum Drop Game using computer vision. Build for fun on top of DXCam (screenshots), OpenCV (object
detection) and Pynput (mouse controller).

## Requirements

- Python 3.12 and higher.
- Any OS (only Windows tested).

## Usage

1. Install dependencies
2. Add `blum_window.png`, `play.png`, `replay.png` screenshots into the `img` directory (need to detect game window,
   play and play again buttons).
3. Open the game. The play button should be visible.
4. Run the script (`python main.py`).
5. To exit script *press/release/scroll* any mouse button during 1s between games.
