import numpy as np
import cv2 as cv

from animation import Animator


class CardOverlayer:
    def __init__(self, animator: Animator, frame_shape: tuple):
        self.frame_shape: tuple = frame_shape
        self.animator = animator
        self.graphics: dict[any, np.ndarray, np.ndarray] = {}

    def set_animator(self, animator: Animator):
        self.animator = animator

    def overlay_cards(self, frame: np.ndarray, found_cards: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        h, w = self.frame_shape[:2]
        modified_frame = np.copy(frame)

        for card, (projected_corners, fill_mask) in found_cards.items():
            # Get activate graphics for the card
            graphics_image = self.animator.get_graphics(card)
            if graphics_image is None:
                continue  # No activate animations, so don't display anything here

            graphics_corners = np.array([[0., 0.],
                                         [1., 0.],
                                         [1., 1.],
                                         [0., 1.]])
            # Adjust for image resolution
            # Shape is (h, w), so reverse
            graphics_size = np.array([graphics_image.shape[1], graphics_image.shape[0]])
            graphics_corners *= graphics_size

            # Resolve perspective of card
            perspective = cv.getPerspectiveTransform(src=graphics_corners.astype(dtype=np.float32),
                                                     dst=projected_corners.astype(dtype=np.float32))

            warped_graphics = cv.warpPerspective(src=graphics_image, M=perspective, dsize=(w, h))

            # Using np.where for result for masking seems to be faster than alternatives...
            # I previously tried using cv.bitwise_and + cv.bitwise_or with a fill and a keep mask
            # But these masks need to be applied across all color channels
            # This means I either need to apply the masks 3 times or stack/repeat them (both of which are slower)
            where_mask = np.where(fill_mask == 255)
            modified_frame[where_mask] = warped_graphics[where_mask]
        return modified_frame
