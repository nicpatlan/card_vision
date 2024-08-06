import argparse
import sys
import time

import numpy as np
import cv2 as cv

import playing_cards
import uno
from animation import Animator
from deck import DetectionDeck, register_cards
from overlayment import CardOverlayer
from segmentation import CardSegmenter


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    feed_group = parser.add_mutually_exclusive_group(required=True)
    feed_group.add_argument('--capture', '--cap', '-i', type=str,
                            help="a live capture stream to process")
    feed_group.add_argument('--source', '--src', '-s', type=str,
                            help="a video file to process")
    parser.add_argument('--camera-parameters', '--params', '-p',
                        type=str, default='params.npz',
                        help=("a .npz file location containing the input "
                              "video's camera calibration parameters."))

    parser.add_argument('--output', '--out', '-o', type=str,
                        help="an output location for the processed video")
    parser.add_argument('--no-preview', '--quiet', '-q', action='store_true',
                        help=("if the video should be processed without "
                              "displaying a preview frame of the result."))

    parser.add_argument('--downscaling', '-d', type=int, default=1)
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--fps', type=float)

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

    # If this is a live feed, or a pre-recorded videos
    live: bool = args.capture is not None
    # Prerecorded videos will be processed frame by frame, without skips
    if live:
        src = args.capture
    else:
        src = args.source

    # Convert if we are passed a video device number
    if live and src.isdigit():
        src = int(src)

    cap = cv.VideoCapture(src)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv.CAP_PROP_FPS))

    # Can potentially scale the input size to reduce needed processing
    input_downscaling: int = args.downscaling
    scaled_width: int = w // input_downscaling
    scaled_height: int = h // input_downscaling

    target_width: int = args.width
    target_height: int = args.height
    target_fps: float = fps if args.fps is None else args.fps
    frame_length: float = 1. / target_fps

    recording = args.output is not None
    if recording:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(filename=args.output, fourcc=fourcc, fps=target_fps,
                             frameSize=(target_width, target_height))

    camera_parameters = np.load(args.camera_parameters, allow_pickle=False)
    camera_matrix: np.ndarray = camera_parameters['matrix']
    # Camera matrix is related to frame size, so scale appropriately
    camera_matrix[0:1] /= input_downscaling
    distortion = camera_parameters['distortion']

    quiet = args.no_preview
    deck = DetectionDeck()

    register_cards(deck, hls=True)

    animator_idx = 1
    animators: list[Animator] = [
        playing_cards.create_animator(
            {card: idx for (idx, card) in enumerate(deck.cards)}
        ), uno.create_animator(
            {card: idx for (idx, card) in enumerate(deck.cards)}
        )]

    segmentor = CardSegmenter(deck=deck,
                              camera_matrix=camera_matrix,
                              distortion=distortion,
                              frame_shape=(scaled_height, scaled_width, 3))

    overlayer = CardOverlayer(animator=animators[animator_idx], frame_shape=(scaled_height, scaled_width, 3))

    if live:
        # Have unbuffered input when live to ensure fresh frames
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    if not quiet:
        cv.namedWindow('frame')
    if live:
        last_frame_time = time.time()
    else:
        last_frame_time = 0.0

    animators[animator_idx].skip_time(last_frame_time)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if live:
            current_time = time.time()
        else:
            frame_count = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            current_time = (frame_count / fps)

        segmentor.garbage_collect_samples(current_time)
        animators[animator_idx].step_time(current_time)

        if input_downscaling != 1:
            scaled_frame: np.ndarray = cv.resize(frame, (scaled_width, scaled_height))
        else:
            scaled_frame = frame

        found_cards = segmentor.segment_frame(scaled_frame, frame_time=current_time)

        modified_image = overlayer.overlay_cards(scaled_frame, found_cards)

        # Resize output to be less than a certain size (in case our source image was too high-res)
        resized_image = resize_with_border(modified_image, target_width, target_height)
        if not quiet:
            cv.imshow('frame', resized_image)

        # Record frame, making as many copies as needed to meet frame rate
        while last_frame_time < current_time:
            last_frame_time += frame_length
            if recording:
                out.write(resized_image)

        if not quiet:
            pressed = cv.waitKey(1)
            if pressed == ord('q'):  # Quit on q
                break
            elif pressed == ord('n'):  # Step to the next animation type
                animator_idx = (animator_idx + 1) % len(animators)
                animators[animator_idx].step_time(current_time)
                overlayer.set_animator(animators[animator_idx])

    if not quiet:
        cv.destroyWindow('frame')
    cap.release()
    if recording:
        out.release()


def resize_with_border(img: np.ndarray, target_width: int, target_height: int):
    height, width = img.shape[:2]
    if target_width == width and target_height == height:
        return img  # Size is unchanged

    ratio: float = min(target_width / width, target_height / height)
    scaled: np.ndarray = cv.resize(img, (0, 0), fx=ratio, fy=ratio)

    scaled_height, scaled_width = scaled.shape[:2]

    horizontal_padding: int = target_width - scaled_width
    vertical_padding: int = target_height - scaled_height

    left_padding: int = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding
    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding

    padded = cv.copyMakeBorder(scaled,
                               top=top_padding, bottom=bottom_padding,
                               left=left_padding, right=right_padding,
                               borderType=cv.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


if __name__ == "__main__":
    main()
