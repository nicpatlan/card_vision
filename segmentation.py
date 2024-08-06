from typing import Callable

import numpy as np
import cv2 as cv

from deck import DetectionDeck


class CardSegmenter:
    def __init__(self, deck: DetectionDeck,
                 camera_matrix: np.ndarray,
                 distortion: np.ndarray,
                 frame_shape: tuple):
        self.deck: DetectionDeck = deck

        self.camera_matrix: np.ndarray = camera_matrix
        self.distortion: np.ndarray = distortion
        self.frame_shape: tuple = frame_shape

        self.color_space_converter: Callable[[np.ndarray], np.ndarray] = to_hsl
        self.blur_size = int(min(frame_shape[0], frame_shape[1]) // 500)

        self.sample_radius: int = int(min(frame_shape[0], frame_shape[1]) // 500)
        self.color_outlier_threshold = np.array([1, 0.1, 0.2])

        self.memory_duration: float = 0.4
        self._projected_corners: dict[str, dict[float, np.ndarray]] = {}
        self._color_samples: dict[str, dict[float, np.ndarray]] = {}

        self.background_kernel_size = int(min(frame_shape[0], frame_shape[1]) // 100)
        self.estimate_kernel_size = int(min(frame_shape[0], frame_shape[1]) // 200)
        self.segmentation_kernel = int(min(frame_shape[0], frame_shape[1]) // 100)
        # Make kernel sizes odd
        self.background_kernel_size += (1 - self.background_kernel_size % 2)
        self.estimate_kernel_size += (1 - self.estimate_kernel_size % 2)
        self.segmentation_kernel += (1 - self.segmentation_kernel % 2)

    def reset_memory(self):
        self._projected_corners = {}
        self._color_samples = {}

    def garbage_collect_samples(self, current_time: float):
        for card in self._projected_corners:
            for time in list(self._projected_corners[card].keys()):
                if current_time - time > self.memory_duration:
                    del self._projected_corners[card][time]

        for card in self._color_samples:
            for time in list(self._color_samples[card].keys()):
                if current_time - time > self.memory_duration:
                    del self._color_samples[card][time]

    def remember_projected_corners(self, card: str, frame_time: float, corners: np.ndarray):
        if card not in self._projected_corners:
            self._projected_corners[card] = {}

        self._projected_corners[card][frame_time] = corners

    def remember_color_samples(self, card: str, frame_time: float, samples: np.ndarray) -> np.ndarray | None:
        if card not in self._color_samples:
            self._color_samples[card] = {}

        self._color_samples[card][frame_time] = samples

    def recall_projected_corners(self, card: str, frame_time: float) -> np.ndarray | None:
        # Based on most recently pushed value and/or past values
        # Smooth/interpolate current corner values
        if card not in self._color_samples:
            return None  # We have no data on this card

        card_corner_samples = self._projected_corners[card]

        if len(card_corner_samples) >= 3:
            time_steps = []
            samples = []
            for time, corners in card_corner_samples.items():
                time_steps.append(np.array([time]))
                samples.append(corners.reshape(8))  # Squash shape for prediction purposes
            time_steps = np.concatenate(time_steps)
            samples = np.stack(samples, axis=0)

            fit: np.ndarray = np.polynomial.polynomial.polyfit(time_steps, samples, deg=2)
            predicted = np.polynomial.polynomial.polyval(frame_time, fit)
            predicted = predicted.reshape((4, 2))  # Un-squash into four 2d points
        else:
            predicted = None

        if frame_time in card_corner_samples:
            # We have recent data, so use it!
            actual = card_corner_samples[frame_time]
        else:
            actual = None

        if actual is None:
            return predicted
        if predicted is None:
            return actual

        alpha = 0.8
        return actual * alpha + predicted * (1 - alpha)

    def recall_color_samples(self, card, frame_time: float) -> np.ndarray | None:
        # Based on most recently pushed value and/or past values
        # Smooth/interpolate current corner values
        if card not in self._color_samples:
            return None  # We have no data on this card

        card_color_samples = self._color_samples[card]
        if frame_time in card_color_samples:
            return card_color_samples[frame_time]  # We have recent data, so use it!

        if len(card_color_samples.keys()) == 0:
            return  # No other samples to draw from

        most_recent_time = max(card_color_samples.keys(), default=None)
        return card_color_samples[most_recent_time]

    def segment_frame(self, frame: np.ndarray, frame_time: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        h, w = self.frame_shape[:2]

        estimate_kernel = np.ones((self.estimate_kernel_size, self.estimate_kernel_size))

        color_image = self.color_space_converter(frame.astype(np.float32) / 255)

        matches = self.deck.find_cards(frame, camera_matrix=self.camera_matrix, distortion=self.distortion)

        card_estimates = np.zeros((h, w), dtype=np.int32)
        net_bounds_mask = np.zeros((h, w), dtype=np.ubyte)
        claims = np.zeros((h, w), dtype=np.ubyte)

        corners: dict[str, np.ndarray] = {}
        bounds_masks: dict[str, np.ndarray] = {}
        for idx, card in enumerate(self.deck.card_keys()):
            if card in matches:
                # If we found the card, recompute fresh projected corners and color samples
                rotation_vector, translation, _, marker_match_points = matches[card]

                card_layout = self.deck.cards[card].layout
                card_corners = np.array([[0., 0., 0.],
                                         [1., 0., 0.],
                                         [1., 1., 0.],
                                         [0., 1., 0.]])  # In object space of the card
                card_corners *= np.array([*card_layout.card_dim, 0])
                card_corners -= card_layout.origin_shift()

                raw_projected_corners, _ = cv.projectPoints(objectPoints=card_corners,
                                                            rvec=rotation_vector, tvec=translation,
                                                            cameraMatrix=self.camera_matrix, distCoeffs=self.distortion)
                # Remove extraneous inner dimension
                raw_projected_corners = raw_projected_corners.reshape((4, 2))
                self.remember_projected_corners(card, frame_time, raw_projected_corners)

                raw_color_samples = _get_expanded_samples(color_image,
                                                          marker_match_points.astype(dtype=np.int32),
                                                          self.sample_radius)
                self.remember_color_samples(card, frame_time, raw_color_samples)

            projected_corners = self.recall_projected_corners(card, frame_time)
            if projected_corners is None:
                continue
            corners[card] = projected_corners

            bounds_mask = np.zeros((h, w), dtype=np.ubyte)
            cv.fillConvexPoly(img=bounds_mask,
                              # Casting points to int, so they can be interpreted as image coordinates
                              points=projected_corners.astype(dtype=np.int32),
                              color=255)
            bounds_masks[card] = bounds_mask

            # Find rough estimates of segmentation based on similar color to known points
            color_samples = self.recall_color_samples(card, frame_time)
            if color_samples is None:
                continue
            hue_mask = _find_similarity_mask(color_image, color_samples,
                                             outlier_threshold=self.color_outlier_threshold)

            # Card is (probably) non-occluded in on pixels of matching hue withing the bounds of its 4 corners
            combined_mask = cv.bitwise_and(bounds_mask, hue_mask)

            # Fill any small holes in the mask
            combined_mask = cv.morphologyEx(combined_mask, op=cv.MORPH_CLOSE, kernel=estimate_kernel, iterations=2)
            combined_mask = cv.morphologyEx(combined_mask, op=cv.MORPH_OPEN, kernel=estimate_kernel, iterations=1)

            # Combine estimates with masks for other cards
            card_estimates[combined_mask > 0] = idx + 2  # Value 0 is for unknown values, 1 is for background
            net_bounds_mask[bounds_mask > 0] = 255  # Areas where there might be a card
            claims[combined_mask > 0] += 1

        background_kernel = np.ones((self.background_kernel_size, self.background_kernel_size))
        # Expand the region where we think the cards could be
        net_bounds_mask = cv.dilate(net_bounds_mask, kernel=background_kernel)
        card_estimates[net_bounds_mask == 0] = 1  # Everything outside of this is probably background
        card_estimates[claims >= 2] = 0  # Contested areas can have accurate no estimates

        color_image[:, : 1:] = cv.blur(color_image[:, : 1:], (self.blur_size, self.blur_size))
        color_image[:, :, [1, 2]] = color_image[:, :, [2, 1]]
        blurred = (cv.cvtColor(color_image, code=cv.COLOR_HLS2BGR) * 255).astype(dtype=np.ubyte)
        segmentations = cv.watershed(blurred, card_estimates)

        segmentation_kernel = np.ones((self.segmentation_kernel, self.segmentation_kernel))

        found: dict[str, (np.ndarray, np.ndarray)] = {}
        for idx, card in enumerate(self.deck.card_keys()):
            if card not in corners:
                continue

            fill_mask = np.zeros((h, w), dtype=np.ubyte)
            fill_mask[segmentations == idx + 2] = 255  # Extract just this card from the image segmentation

            # Fill gaps inside the card
            fill_mask = cv.morphologyEx(fill_mask, op=cv.MORPH_CLOSE, kernel=segmentation_kernel)

            # Smooth the edges
            fill_mask = cv.morphologyEx(fill_mask, op=cv.MORPH_OPEN, kernel=segmentation_kernel, iterations=1)

            # Clamp to bounds if watershed accidentally went over
            fill_mask = cv.bitwise_and(bounds_masks[card], fill_mask)
            found[card] = (corners[card], fill_mask)
        return found


def to_hsv(image: np.ndarray) -> np.ndarray:
    return cv.cvtColor(image, code=cv.COLOR_BGR2HSV)


def to_hsl(image: np.ndarray) -> np.ndarray:
    hsl = cv.cvtColor(image, code=cv.COLOR_BGR2HLS)
    # Swapping channel order to be consistent with hsv
    hsl[:, :, [1, 2]] = hsl[:, :, [2, 1]]
    return hsl


def _get_expanded_samples(image: np.ndarray, points, radius: int = 0) -> np.ndarray:
    h, w, c = image.shape
    return np.concatenate([image[max(0, pt[1] - radius):min(h, pt[1] + radius + 1),
                                 max(0, pt[0] - radius):min(w, pt[0] + radius + 1)]
                          .reshape((-1, c)) for pt in points.astype(dtype=np.int32)],
                          axis=0)


def _find_similarity_mask(image: np.ndarray, samples: np.ndarray, outlier_threshold: np.ndarray = None) -> np.ndarray:
    """
    :param image: HxWxC array of an image
    :param samples: Nx2 array of points in the original image to match the color from
    :param outlier_threshold: how far above or below the seen range a pixel can be before exclusion from the mask
    :return: HxW bitmask of regions with matching color (stored using unsigned bytes)
    """
    if outlier_threshold is None:
        outlier_threshold = np.zeros(image.shape[-1])
    low = np.percentile(samples, q=5, axis=0) - outlier_threshold
    high = np.percentile(samples, q=95, axis=0) + outlier_threshold
    mask = cv.inRange(image, low, high)
    return mask
