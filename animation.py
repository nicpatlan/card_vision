import numpy as np
import cv2 as cv


class Animator():
    def skip_time(self, time: float):
        pass

    def step_time(self, time: float):
        pass

    def get_graphics(self, card: str) -> np.ndarray:
        pass


class BouncingAnimator(Animator):
    def __init__(self, length: float = 1., scale: float = 0.5, mean_delay: float = 2.):
        self.backgrounds: dict[str, np.ndarray] = {}
        self.foregrounds: dict[str, np.ndarray] = {}
        self.bounce_states: dict[str, float] = {}
        self.bounce_length: float = length
        self.min_scale: float = scale
        self.mean_bounce_delay: float = mean_delay
        self.last_time_step: float = 0.
        self.rng = np.random.default_rng(455)

    def set_animation(self, card: str, background: np.ndarray, foreground: np.ndarray):
        assert foreground.shape[2] == 4  # Foreground should have transparency
        self.backgrounds[card] = background
        self.foregrounds[card] = foreground
        self.bounce_states[card] = 0.

    def skip_time(self, time: float):
        self.last_time_step = time

    def step_time(self, time):
        delta = time - self.last_time_step
        self.last_time_step = time
        for card in self.bounce_states.keys():
            self.bounce_states[card] += delta
            if self.bounce_states[card] > self.bounce_length:
                time_to_next_bounce = self.rng.exponential(self.mean_bounce_delay)
                self.bounce_states[card] -= (self.bounce_length + time_to_next_bounce)

    def get_graphics(self, card: str) -> np.ndarray | None:
        if card not in self.bounce_states:
            return None
        background = self.backgrounds[card]
        foreground = self.foregrounds[card]
        #cv.imshow('test', foreground[:, :, 3])
        #cv.waitKey(0)

        state = self.bounce_states[card]

        # [0, 1] portion of animation length
        animation_point = state / self.bounce_length

        # 0 for no scaling, 1 for max scaling
        intensity: float = 1 - abs(2 * animation_point - 1)
        scale: float = min(1., max(0., (1 - intensity) + self.min_scale * intensity))

        if scale == 1: # Unchanged
            padded_foreground = np.copy(foreground)
        else:
            height, width, c = foreground.shape
            scaled_height = int(height * scale)
            scaled_width = int(width * scale)
            scaled_foreground = cv.resize(foreground, (scaled_width, scaled_height))

            horizontal_padding: int = width - scaled_width
            vertical_padding: int = height - scaled_height

            left_padding: int = horizontal_padding // 2
            right_padding = horizontal_padding - left_padding
            top_padding = vertical_padding // 2
            bottom_padding = vertical_padding - top_padding

            padded_foreground = cv.copyMakeBorder(scaled_foreground,
                                                  top=top_padding, bottom=bottom_padding,
                                                  left=left_padding, right=right_padding,
                                                  borderType=cv.BORDER_CONSTANT, value=(0, 0, 0, 0))

        merged = np.copy(background)
        mask = np.where(padded_foreground[:, :, 3] > 0)

        padded_foreground = padded_foreground[:, :, :-1]
        merged[mask] = padded_foreground[mask]
        return merged


class StaticAnimator(Animator):
    def __init__(self):
        self.activate: dict[str, np.ndarray] = {}

    def set_animation(self, card: str, graphics: np.ndarray):
        self.activate[card] = graphics

    def skip_time(self, time: float):
        pass

    def step_time(self, time: float):
        pass

    def get_graphics(self, card: str) -> np.ndarray | None:
        if card not in self.activate:
            return None
        return self.activate[card]

