import cv2 as cv
import numpy as np

from animation import Animator, BouncingAnimator


def load_graphics() -> list[tuple[np.ndarray, np.ndarray]]:
    sheet: np.ndarray = cv.imread('uno/cards_sheet.png')
    mask: np.ndarray = cv.imread('uno/mask.png')
    if sheet is None or mask is None:
        raise FileNotFoundError()
    sheet_width = sheet.shape[1]
    mask = mask[:, :, 0]
    card_width = mask.shape[1]
    card_height = mask.shape[0]

    images: list[tuple[np.ndarray, np.ndarray]] = []
    for c in range(sheet_width//card_width):
        card_bg = np.copy(sheet[:, c*card_width:(c+1)*card_width, :])
        card_fg = np.zeros((card_height, card_width, 4), dtype=np.ubyte)
        card_fg[:, :, :-1][mask > 0] = card_bg[mask > 0]
        card_fg[:, :, :-1][mask == 0] = 255
        card_fg[:, :, -1][mask > 0] = 255
        card_bg[mask > 0] = 255
        images.append((card_bg, card_fg))
    return images


def create_animator(card_assignments: dict[str, int]) -> Animator:
    images = load_graphics()
    animator = BouncingAnimator()
    for (card, assignment) in card_assignments.items():
        bg, fg = images[assignment]
        animator.set_animation(card, background=bg, foreground=fg)
    return animator
