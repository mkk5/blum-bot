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


def find_objects(hsv_img: np.ndarray, hsv_ranges: Sequence[HSVRange]) -> Sequence[np.ndarray]:
    """Find objects by color using combined HSV color mask, return contour should be additionally filtered by area."""
    masks = (cv2.inRange(hsv_img, np.array(hsv_range[0]), np.array(hsv_range[1])) for hsv_range in hsv_ranges)
    combined_mask = reduce(cv2.bitwise_or, masks)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def centers_by_area(contour: np.ndarray, min_area: int) -> Generator[Point, None, None]:
    """Filter contour by min_area and return contour center point."""
    m = cv2.moments(contour)
    if m['m00'] > min_area:
        x_center = int(m['m10'] / m['m00'])
        y_center = int(m['m01'] / m['m00'])
        yield x_center, y_center

WHITE = ((0, 0, 210), (180, 110, 255))
PINK_RED = ((143, 80, 80), (180, 255, 255))
GREEN = ((40, 80, 80), (55, 255, 255))
BROWN_YELLOW = ((10, 80, 80), (30, 255, 255))

def process_image(img: np.ndarray) -> Generator[Point, None, None]:
    """Returns points to click."""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for contour in find_objects(hsv_img, [WHITE, PINK_RED, GREEN, BROWN_YELLOW]):
        yield from centers_by_area(contour, 210)


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
