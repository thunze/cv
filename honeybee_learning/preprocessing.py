"""Data preprocessing script.

Extracts bee crops from video frames based on trajectory information and saves them
as two numpy arrays: One for the crops and one for the metadata.

We're explicitly saving the crops as one large binary file instead of multiple images
because RAMSES seems to throttle I/O operations on `/scratch` when handling many
small files, which previously made the process of loading crops for training very
slow.
"""

from __future__ import annotations

import logging
import os
import traceback

import cv2
import numpy as np

from .config import CROPS_PATH, METADATA_PATH

__all__ = ["main"]


def extract_trajectory_information(trajectories_path):
    """Given a path to a folder containing `.txt` files with trajectory information,
    extract this information and save it in a dictionary.

    Args:
        trajectories_path: Path to the folder containing the trajectory files.

    Returns:
        A dictionary containing detection information, grouped by
        `(recording_no, frame_no)`. Individual bees are stored in a dictionary
        containing `bee_no`, `pos_x`, `pos_y`, `class_no` and `angle`.
    """

    # Dictionary to store detections grouped by (recording_no, frame_no)
    detections_by_frame = {}

    for root, _, files in os.walk(trajectories_path):
        for file in files:
            if file.endswith(".txt"):
                rec_no = root[-1]
                bee_no = file.replace(".txt", "")
                file_path = os.path.join(root, file)
                try:
                    # Open .txt file and extract all information
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line_parts = line.strip().split(",")
                            frame_no = int(line_parts[0])
                            pos_x = int(line_parts[1])
                            pos_y = int(line_parts[2])
                            class_no = line_parts[3]
                            angle = line_parts[4]

                            key = (rec_no, frame_no)
                            if key not in detections_by_frame:
                                detections_by_frame[key] = []
                            detections_by_frame[key].append(
                                {
                                    "bee_no": bee_no,
                                    "pos_x": pos_x,
                                    "pos_y": pos_y,
                                    "class_no": class_no,
                                    "angle": angle,
                                }
                            )
                except Exception as e:
                    logging.exception(f"Error reading {file_path}: {e}")
    return detections_by_frame


def extract_frames_from_video(videos_path, frames_path):
    """Extract all frames from all videos in the given `video_path` and save them to
    `frames_path`.

    In case the folder for that recording already exists, that video gets skipped.

    Args:
        videos_path: Path to the folder containing the videos.
        frames_path: Path to the folder the frame images are supposed to be saved in.
    """
    for root, _, files in os.walk(videos_path):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                rec_no = file.replace(".mp4", "")[-1]

                # If folder already exists, assume that all frames were successfully
                # extracted, might change this later to be more robust
                if os.path.isdir(frames_path + "rec" + rec_no):
                    logging.info(f"Skipping video {file}")
                else:
                    os.makedirs(frames_path + "rec" + rec_no, exist_ok=True)
                    vidcap = cv2.VideoCapture(video_path)
                    count = 0
                    success = True
                    while success:
                        try:
                            success, image = vidcap.read()
                            if success and image is not None:
                                cv2.imwrite(
                                    f"{frames_path}/rec{rec_no}/frame{count:04d}.png",
                                    image,
                                )
                                count += 1
                            else:
                                break
                        except Exception as e:
                            logging.exception(
                                "Exception while extracting frames: "
                                + traceback.format_exc(e)
                            )
                logging.info("Finished extracting frames for recording " + rec_no)


def crop_bee(img, y, x, crop_size=256):
    """Crop a square region of size `crop_size` x `crop_size` centered at `(y, x)`
    from a grayscale image.

    Moves the crop window if necessary to stay within bounds.

    Note that x and y coordinates are swapped because of mislabeling in the dataset,
    so `x` is the horizontal coordinate and `y` is the vertical coordinate.

    Args:
        img: The input image.
        y: X-coordinate of the center point.
        x: Y-coordinate of the center point.
        crop_size (int): Size of the crop (default is 256).

    Returns:
        Cropped image of shape `(crop_size, crop_size)`.
    """
    if img is None:
        raise ValueError("Image not found or failed to load.")

    height, width = img.shape
    half = crop_size // 2

    # Initial crop boundaries
    start_x = x - half
    start_y = y - half
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    # Adjust crop window if it goes out of bounds
    if start_x < 0:
        start_x = 0
        end_x = crop_size
    elif end_x > width:
        end_x = width
        start_x = width - crop_size

    if start_y < 0:
        start_y = 0
        end_y = crop_size
    elif end_y > height:
        end_y = height
        start_y = height - crop_size

    # Final crop
    crop = img[start_y:end_y, start_x:end_x]

    return crop


def extract_save_crops(bee_detections_by_frame, frames_path, bee_number=None):
    """Given a dictionary of bee detections, go through all frames and crop all bees
    for that frame.

    If `bee_number` is specified, only extracts crops for that bee.

    Args:
        bee_detections_by_frame: Dictionary of bee detections.
        frames_path: Path to folder containing the frames.
        bee_number: Optional, to extract crops for only one bee. For testing purposes.
    """
    crops = []
    metadata = []

    # Loop through all (rec_no, frame_no) pairs and extract all (or the single one)
    # bees for that frame, so frames only have to be opened once each
    for (rec_no, frame_no), bees in sorted(bee_detections_by_frame.items()):
        frame_name = f"frame{str(frame_no).rjust(4, '0')}.png"
        frame_path = os.path.join(frames_path, f"rec{rec_no}", frame_name)
        frame_img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        if frame_img is None:
            logging.info(f"Warning: Could not load frame {frame_path}")
            continue

        # logging.info(f"{len(bees)} bees found in frame {frame_no}, rec {rec_no}.")
        for bee in bees:
            if bee_number is not None and bee["bee_no"] != bee_number:
                continue  # Skip bees that are not the target one

            pos_x = bee["pos_x"]
            pos_y = bee["pos_y"]
            bee_no = bee["bee_no"]
            class_no = bee["class_no"]
            angle = bee["angle"]

            # Get crop and save it
            # logging.info(
            #     f"Cropping bee no. {bee_no}, rec no. {rec_no}, frame no. {frame_no}"
            # )
            crop = crop_bee(frame_img, pos_x, pos_y, crop_size=128)

            # Resize crop to (224,224)
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            crops.append(resized)

            metadata.append(
                (int(rec_no), int(frame_no), int(bee_no), int(class_no), int(angle))
            )

        logging.info(
            f"Finished croppings for: Recording no. {rec_no}, frame no. {frame_no} "
        )

    crops_array = np.array(crops, dtype=np.uint8)
    metadata_array = np.array(metadata, dtype=np.uint16)

    return crops_array, metadata_array


def main():
    """Main function to extract crops from bee detection trajectories."""
    # Initialize logging
    logging.basicConfig(
        filename="/scratch/cv-course2025/group7/processing128.log",  # Save logs to file
        level=logging.INFO,  # Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    )

    logging.info("Starting trajectory extraction")
    trajectories_path = "/scratch/cv-course2025/group7/trajectories/"
    vid_path = "/scratch/cv-course2025/group7/videos/"
    frames_path = "/scratch/cv-course2025/group7/frames/"

    logging.info("Starting trajectory extraction")
    bee_detections_by_frame = extract_trajectory_information(trajectories_path)

    logging.info(
        "Trajectory extraction finished, starting frame extraction from videos"
    )
    extract_frames_from_video(vid_path, frames_path)
    logging.info("Frame extraction finished.")

    logging.info("Starting cropping.")
    crops_array, metadata_array = extract_save_crops(
        bee_detections_by_frame, frames_path
    )

    # Make sure that `CROPS_PATH` exists
    os.makedirs(CROPS_PATH, exist_ok=True)

    np.save(CROPS_PATH, crops_array)
    np.save(METADATA_PATH, metadata_array)

    logging.info("Program execution done.")


if __name__ == "__main__":
    main()
