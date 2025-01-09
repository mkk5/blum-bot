# Blum bot

Performant clicker for Blum Drop Game using computer vision. Build for fun on top of DXCam (screenshots), OpenCV (object
detection) and Pynput (mouse controller).

## Requirements

- `uv` installed

## Usage

1. Add `blum_window.png`, `play.png`, `replay.png` screenshots into the `img` directory (need to detect game window,
   play and play again buttons).
2. Open the game. The play button should be visible.
3. Run the script (`uv run main.py`).
4. To exit script *press/release/scroll* any mouse button during 1s between games.
