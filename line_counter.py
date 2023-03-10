from typing import Dict
from supervision.geometry.dataclasses import Point, Rect, Vector
from supervision.tools.detections import Detections

from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Point:
    x: float
    y: float

    def as_xy_int_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)

    def as_xy_float_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

@dataclass
class Vector:
    start: Point
    end: Point

    def is_in(self, point: Point) -> bool:
        v1 = Vector(self.start, self.end)
        v2 = Vector(self.start, point)
        cross_product = (v1.end.x - v1.start.x) * (v2.end.y - v2.start.y) - (
            v1.end.y - v1.start.y
        ) * (v2.end.x - v2.start.x)
        return cross_product < 0

@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    def pad(self, padding) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )


class LineCounter:
    def __init__(self, start, end):
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, bool] = {}
        self.in_count = 0
        self.out_count = 0

    def update(self, detections: Detections):
        for xyxy, confidence, class_id, tracker_id in detections:
            if tracker_id is None:
                continue

            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2)
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            # detection is partially in and partially out
            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]
            # handle new detection
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            # handle detection on the same time side of the line
            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            if tracker_state: self.in_count += 1 # type: ignore
            else: self.out_count += 1 # type: ignore


class LineCounterAnnotator:
    def __init__(
            self,
            thickness: float = 2,
            color: str = 'black',  # white
            text_thickness: float = 2,
            text_color: str = 'white', # black
            text_scale: float = 0.5,
            text_offset: float = 1.5,
            text_padding: int = 10
    ):
        self.thickness: float = thickness
        self.color: tuple = self._color(color)
        self.text_thickness: float = text_thickness
        self.text_color: tuple = self._color(text_color)
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding


    def _color(self, color) -> tuple:
        c = {
            "black": (0,0,0),
            "white": (255, 255, 255)
        }
        return c[color]

    def annotate(self, frame: np.ndarray, line_counter: LineCounter) -> np.ndarray:
        cv2.line(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            line_counter.vector.end.as_xy_int_tuple(),
            self.color.as_bgr(),
            self.thickness,
            lineType=cv2.LINE_AA,
            shift=0
        )
        cv2.circle(
            frame,
            line_counter.vector.start.as_xy_int_tuple(),
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            line_counter.vector.end.as_xy_int_tuple(),
            radius=5,
            color=self.text_color.as_bgr(),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        in_text = f"in: {line_counter.in_count}"
        out_text = f"in: {line_counter.out_count}"

        (in_text_width, in_text_height), _ = cv2.getTextSize(
            in_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        (out_text_width, out_text_height), _ = cv2.getTextSize(
            out_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        in_text_x = int(
            (line_counter.vector.end.x + line_counter.start.x - in_text_width) / 2
        )

        in_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + in_text_height)
            / 2
            - self.text_offset * in_text_height
        )
        out_text_x = int(
            (line_counter.vector.end.x + line_counter.vector.start.x - out_text_width)
            / 2
        )
        out_text_y = int(
            (line_counter.vector.end.y + line_counter.vector.start.y + out_text_height)
            / 2
            + self.text_offset * out_text_height
        )


        in_text_background_rect = Rect(
            x=in_text_x,
            y=in_text_y - in_text_height,
            width=in_text_width,
            height=in_text_height,
        ).pad(padding=self.text_padding)
        out_text_background_rect = Rect(
            x=out_text_x,
            y=out_text_y - out_text_height,
            width=out_text_width,
            height=out_text_height,
        ).pad(padding=self.text_padding)

        cv2.rectangle(
            frame,
            in_text_background_rect.top_left.as_xy_int_tuple(),
            in_text_background_rect.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )
        cv2.rectangle(
            frame,
            out_text_background_rect.top_left.as_xy_int_tuple(),
            out_text_background_rect.bottom_right.as_xy_int_tuple(),
            self.color.as_bgr(),
            -1,
        )
        cv2.putText(
            frame,
            in_text,
            (in_text_x, in_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            out_text,
            (out_text_x, out_text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.text_color.as_bgr(),
            self.text_thickness,
            cv2.LINE_AA,
        )
