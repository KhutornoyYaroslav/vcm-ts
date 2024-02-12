import cv2 as cv


def get_video_length(video_path: str, countable: int = False):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file '{video_path}'")
        return -1

    if countable:
        count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    else:
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1

    return int(count)


def get_video_resolution(video_path: str):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file '{video_path}'")
        return -1

    w, h = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, c = frame.shape
        break

    return w, h
