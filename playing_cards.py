import glob

import cv2 as cv
import numpy as np

from animation import Animator, StaticAnimator


def load_graphics() -> list[np.ndarray]:
    image_paths = glob.glob('playing_cards/*.png')
    images = []
    for image_path in image_paths:
        images.append(cv.imread(image_path))
    return images


def create_animator(card_assignments: dict[str, int]) -> Animator:
    images = load_graphics()
    animator = StaticAnimator()
    for (card, assignment) in card_assignments.items():
        animator.set_animation(card, images[assignment])
    return animator
