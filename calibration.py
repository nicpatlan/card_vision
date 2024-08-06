import argparse
import cv2 as cv
import numpy as np


class CameraCalibrator:
    def __init__(self, grid_shape: (int, int) = (10, 6),
                 square_length: float = 1, marker_length: float = 0.5):
        self.dictionary: cv.aruco.Dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        self.board: cv.aruco.CharucoBoard = cv.aruco.CharucoBoard(grid_shape, square_length,
                                                                  marker_length, self.dictionary)
        self.detector: cv.aruco.CharucoDetector = cv.aruco.CharucoDetector(self.board)

    def get_grid_shape(self):
        return self.board.getChessboardSize()

    def get_square_length(self) -> float:
        return self.board.getSquareLength()

    def get_marker_length(self) -> float:
        return self.board.getMarkerLength()

    def get_reference_image(self, pixels_per_unit) -> np.ndarray:
        shape = self.get_grid_shape()
        length = self.get_square_length()
        assert length * pixels_per_unit > 1
        size = (int(shape[0] * length * pixels_per_unit),
                int(shape[1] * length * pixels_per_unit))
        return self.board.generateImage(size)

    def calibrate(self, calibration_images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Find corners of calibration boards in provided images
        all_corners = []
        all_ids = []
        for image in calibration_images:
            corners, ids, _, _ = self.detector.detectBoard(image)
            if corners is None or corners.shape[0] < 4:
                continue
            all_corners.append(corners)
            all_ids.append(ids)
        pass

        h, w, c = calibration_images[0].shape
        initial_camera_matrix = np.array([[1000., 0., w / 2.],
                                          [0.,    0., h / 2.],
                                          [0.,    0.,     1.]])
        initial_distortion_coefficients = np.zeros((5, 1))

        # Refine initial camera estimate using found corners
        (retval, camera_matrix,
         distortion_coefficients,
         _, _) = cv.aruco.calibrateCameraCharuco(charucoCorners=all_corners, charucoIds=all_ids,
                                                 board=self.board, imageSize=(h, w),
                                                 cameraMatrix=initial_camera_matrix,
                                                 distCoeffs=initial_distortion_coefficients)
        return camera_matrix, distortion_coefficients


def record_frames_from_stream(stream: cv.VideoCapture, save_limit: int) -> list[np.ndarray]:
    """
    :param stream: A video stream from which frames can be captured; frames will be saved on press of q
    :param save_limit: The maximum number of frames to save; if negative, then frames will be saved until q is pressed
    :return: A list of saved images
    """
    old_buffer_size = stream.get(cv.CAP_PROP_BUFFERSIZE)
    # Switch to only store latest in order to keep output in realtime
    stream.set(cv.CAP_PROP_BUFFERSIZE, 1)

    display_width, display_height = (800, 450)  # Just a very small preview is needed

    save_unlimited = save_limit < 0
    cv.namedWindow('video_stream')
    saved_frames = []
    key_released = False
    while stream.isOpened() and (save_unlimited or len(saved_frames) < save_limit):
        ret, frame = stream.read()
        if not ret:
            break

        # Resize image to fit within desired display bounds
        h, w, c = frame.shape
        ratio = min(display_width / w, display_height / h)
        resized_image = cv.resize(frame, (0, 0), fx=ratio, fy=ratio)

        cv.imshow('video_stream', resized_image)

        pressed = cv.waitKey(1)
        if pressed == ord('q'):  # Quit on q
            print(f"Aborting frame capture")
            break

        # If we just pressed down c, then save the current frame
        if pressed == ord('s') and key_released:
            saved_frames.append(frame)
            if save_unlimited:
                print(f"Captured frame {len(saved_frames)}")
            else:
                print(f"Captured frame {len(saved_frames)}/{save_limit}")
            key_released = False
        else:
            key_released = True

    # Clean up the window
    cv.destroyWindow('video_stream')
    # Restore previous buffer size
    stream.set(cv.CAP_PROP_BUFFERSIZE, old_buffer_size)
    return saved_frames


def generate(args) -> int:
    calibrator = CameraCalibrator()
    ref_img = calibrator.get_reference_image(args.ppi)
    cv.imwrite(args.filename, ref_img)
    return 0


def calibrate(args) -> int:
    if args.capture is not None:
        # Get calibration images from the provided video stream
        stream = cv.VideoCapture()
        if args.capture.isdigit():
            source = int(args.capture)
        else:
            source = args.capture
        stream.open(source)
        limit = -1 if args.count is None else args.count
        frames = record_frames_from_stream(stream, save_limit=limit)
        stream.release()
    else:
        # Get calibration images from provided files
        frames: list[np.ndarray] = []
        for filename in args.files:
            frame = cv.imread(filename)
            if frame is not None:
                frames.append(frame)
            else:  # File might not have existed
                print(f'Could not read \"{filename}\" as an image!')

    if len(frames) == 0:
        print('No images provided!')
        return 1

    calibrator = CameraCalibrator()
    camera_matrix, distortion_coefficients = calibrator.calibrate(calibration_images=frames)
    if args.out is not None:
        np.savez(args.out, matrix=camera_matrix, distortion=distortion_coefficients)
    print(camera_matrix)
    print(distortion_coefficients)
    return 0


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser_generate = subparsers.add_parser('generate')
    parser_generate.add_argument('filename', default='reference_image.png')
    parser_generate.add_argument('--ppi', type=float, default=150)
    parser_generate.set_defaults(func=generate)

    parser_calibrate = subparsers.add_parser('calibrate')
    parser_calibrate.add_argument_group()
    parser_calibrate.add_argument('--out', '-o')
    source_group = parser_calibrate.add_mutually_exclusive_group(required=True)
    source_group.add_argument('files', nargs='*', default=[])
    source_group.add_argument('--capture', '-i', type=str)
    parser_calibrate.add_argument('--count', '-n', type=int)
    parser_calibrate.set_defaults(func=calibrate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    exit(main())
