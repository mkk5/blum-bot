# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dxcam",
#     "opencv-python",
#     "pynput",
# ]
# [tool.uv]
# exclude-newer = "2025-01-09T00:00:00Z"
# ///

import cv2
import numpy as np
import dxcam
from pynput.mouse import Button, Controller, Events
from functools import reduce
import time
from typing import Generator, Sequence


type Point = tuple[int, int]
type Box = tuple[int, int, int, int] # x, y, x+w, y+h | Left-top right-bottom points | DXCam region format
type HSVValue = tuple[int, int, int]
type HSVRange = tuple[HSVValue, HSVValue]


def locate(img: np.ndarray, template: np.ndarray, min_match: float, offset: Point = (0,0)) -> Box | None:
    """Locate template box on image. If minimal similarity is lower than min_match return None."""
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < min_match:
        return None
    x0, y0 = tuple(loc+off for loc, off in zip(max_loc, offset))
    h, w, *_ = template.shape
    return x0, y0, x0+w, y0+h


def locate_on_screen(template: np.ndarray, min_match: float, screen: dxcam.DXCamera, region: Box | None = None) -> Box | None:
    screenshot = screen.grab(region=region)
    if screenshot is None:
        return None
    offset = (0, 0) if region is None else (region[0], region[1])
    return locate(screenshot, template, min_match, offset)


def center(box: Box) -> Point:
    return int((box[0]+box[2]) / 2), int((box[1]+box[3]) / 2)


def click(mouse: Controller, point: Point, offset: Point = (0,0)):
    x, y = tuple(p+off for p, off in zip(point, offset))
    mouse.position = (x, y)
    mouse.press(Button.left)
    mouse.release(Button.left)


def find_objects(hsv_img: np.ndarray, hsv_ranges: Sequence[HSVRange], min_area: int) -> Sequence[np.ndarray]:
    """Find objects using HSV color masks and filter them."""
    masks = (cv2.inRange(hsv_img, np.array(hsv_range[0]), np.array(hsv_range[1])) for hsv_range in hsv_ranges)
    combined_mask = reduce(cv2.bitwise_or, masks)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return filtered_contours


def calc_center(contour: np.ndarray) -> Point:
    """Return contour center point."""
    m = cv2.moments(contour)
    x_center = int(m['m10'] / m['m00'])
    y_center = int(m['m01'] / m['m00'])
    return x_center, y_center

POINTS_RANGE: HSVRange = ((32, 65, 100), (56, 255, 255)) # Green (star), area 200
ICE_RANGE: HSVRange = ((45, 0, 215), (105, 165, 255)) # Blue (ice), area 200
BOMB_RANGE: HSVRange = ((0, 0, 140), (180, 36, 255)) # Gray (bomb), area 150

def process_image(img: np.ndarray) -> Generator[Point, None, None]:
    """Returns points to click."""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for contour in find_objects(hsv_img, [POINTS_RANGE], min_area=200):
        yield calc_center(contour)


def main():
    scn_grabber = dxcam.create(output_color="BGR")
    mouse = Controller()

    blum_window = locate_on_screen(cv2.imread("img/blum_window.png"), 0.8, scn_grabber)
    play_button = locate_on_screen(cv2.imread("img/play.png"), 0.9, scn_grabber, region=blum_window)
    if blum_window is None or play_button is None:
        raise ValueError("Blum window or play button was not found.")
    play_area_box = blum_window[0], blum_window[1]+120, blum_window[2], blum_window[3]-500 # -50 full
    play_button_point = center(play_button)
    click(mouse, play_button_point)

    replay_img = cv2.imread("img/replay.png")
    while True:
        while (replay_button:=locate_on_screen(replay_img, 0.9, scn_grabber, region=blum_window)) is None:
            screenshot = scn_grabber.grab(region=play_area_box)
            if screenshot is not None:
                for point in process_image(screenshot):
                    click(mouse, point, offset=(play_area_box[0], play_area_box[1]))

        with Events() as e:
            event = e.get(1) # Wait for any mouse press/release/scroll event to exit loop between games
            if event is not None: break

        replay_button_point = center(replay_button)
        click(mouse, replay_button_point)
        time.sleep(0.5) # Wait game screen to load and replay button to disappear


if __name__ == '__main__':
    main()
