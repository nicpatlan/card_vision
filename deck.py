from dataclasses import dataclass

import cv2 as cv
import numpy as np
import pathlib

from typing import Dict


@dataclass
class CardLayout:
    card_dim: (float, float) = (2.5, 3.5)
    grid_shape: (int, int) = (4, 6)
    min_padding: float = 0.10

    def marker_length(self) -> float:
        square_length = min((self.card_dim[0] - self.min_padding) / self.grid_shape[0],
                            (self.card_dim[1] - self.min_padding) / self.grid_shape[1])
        return square_length - self.min_padding

    def origin_shift(self) -> np.ndarray:
        square_length = min((self.card_dim[0] - self.min_padding) / self.grid_shape[0],
                            (self.card_dim[1] - self.min_padding) / self.grid_shape[1])
        out = np.array([*self.card_dim, 0]) - np.array([*self.grid_shape, 0], dtype=np.float32) * square_length
        out *= 0.75
        return out


@dataclass
class ColorScheme:
    hue: float = 0.  # Range 0 - 360
    saturation: float = 0.  # 0 - 1
    min_value: float = 0  # 0 - 1
    max_value: float = 1.  # 0 - 1
    hls: bool = False  # Use HLS instead of HSV
# In HLS, min value of 0.2 and max value of 0.6 works well?


@dataclass
class DetectionCard:
    layout: CardLayout
    board: cv.aruco.Board
    color: ColorScheme


class DetectionDeck:
    def __init__(self, dictionary: cv.aruco.Dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)):
        self.dictionary: cv.aruco.Dictionary = dictionary
        self.unused_ids = {i for i in range(len(self.dictionary.bytesList))}
        params = cv.aruco.DetectorParameters()
        #params.useAruco3Detection = False
        self.detector: cv.aruco.ArucoDetector = cv.aruco.ArucoDetector(self.dictionary, params)
        self.cards: dict[str, DetectionCard] = {}

    def register_card(self, key, layout: CardLayout = CardLayout(), color: ColorScheme = ColorScheme()):
        ids = np.array([self.unused_ids.pop() for _ in range(layout.grid_shape[0] *
                                                             layout.grid_shape[1])])
        board: cv.aruco.Board = cv.aruco.GridBoard(layout.grid_shape,
                                                   layout.marker_length(), layout.min_padding,
                                                   self.dictionary, ids)
        self.cards[key] = DetectionCard(layout, board, color)

    def generate_card_image(self, key, pixels_per_unit) -> np.ndarray:
        card: DetectionCard = self.cards[key]
        image_size = (int(card.layout.card_dim[0] * pixels_per_unit),
                      int(card.layout.card_dim[1] * pixels_per_unit))
        margin_size = int(card.layout.min_padding * pixels_per_unit)
        image = card.board.generateImage(outSize=image_size, marginSize=margin_size)
        image = image / 255.  # Convert greyscale image from 0-255 byte range into 0 - 1 float range
        if card.color.hls:
            hsv_image = np.stack([card.color.hue * np.ones(image.shape),
                                  (card.color.max_value - card.color.min_value) * image + card.color.min_value,
                                  card.color.saturation * np.ones(image.shape)],
                                 axis=2, dtype=np.float32)
            return cv.cvtColor(hsv_image, cv.COLOR_HLS2BGR)
        else:  # Use HSV otherwise
            hsv_image = np.stack([card.color.hue * np.ones(image.shape),
                                  card.color.saturation * np.ones(image.shape),
                                  (card.color.max_value - card.color.min_value) * image + card.color.min_value],
                                 axis=2, dtype=np.float32)
            return cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    def find_cards(self, image, camera_matrix, distortion=None) -> dict[str, tuple[np.ndarray, np.ndarray,
                                                                                   np.ndarray, np.ndarray]]:
        corners, ids, rejects = self.detector.detectMarkers(image)
        matches = {}
        for key, card in self.cards.items():
            object_points, image_points = card.board.matchImagePoints(corners, ids)
            if image_points is None or len(image_points) < 4:
                continue  # Couldn't see enough points for the card
            # Find the orientation of the card relative to the camera
            found, rotation_vector, translation = cv.solvePnP(object_points, image_points,
                                                              cameraMatrix=camera_matrix,
                                                              distCoeffs=distortion)
            if not found:
                continue
            matches[key] = (rotation_vector, translation,
                            # Reshape arrays to sensible dimensions (no extra inner dim)
                            object_points.reshape((-1, 3)),
                            image_points.reshape((-1, 2)))
        return matches

    def card_keys(self):
        return self.cards.keys()


def save_img(filename: str, img: np.ndarray):
    if np.issubdtype(img.dtype, np.floating):
        converted = (255 * img).astype(dtype=np.ubyte)
    else:
        converted = img
    saved = cv.imwrite(filename, converted)
    if not saved:
        print(f'Failed to save {filename}!')


def register_cards(deck: DetectionDeck,
                   min_hue: int = 45,
                   max_hue: int = 335,
                   hue_separation: int = 10,
                   hls: bool = False):
    for idx, hue in enumerate(range(min_hue, max_hue + 1, hue_separation)):
        deck.register_card(f'card{idx}', color=ColorScheme(hue=hue, saturation=1,
                                                           min_value=0.45 if hls else 0.4,
                                                           max_value=0.95 if hls else 1.0))


def generate_card_images(deck: DetectionDeck, pixels_per_unit: int = 150, hls: bool = False):
    directory = pathlib.Path(f'boards/{"hls" if hls else "hsv"}')
    if not directory.exists():
        directory.mkdir()
    for card_key in deck.card_keys():
        img = deck.generate_card_image(card_key, pixels_per_unit=pixels_per_unit)
        save_img(f'boards/{"hls" if hls else "hsv"}/{card_key}.png', img)


def main():
    deck = DetectionDeck()
    register_cards(deck)
    generate_card_images(deck, hls=True)


if __name__ == "__main__":
    main()
