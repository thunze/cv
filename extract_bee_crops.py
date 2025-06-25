import os
import cv2






def extract_trajectory_information(trajectories_path):
    """
    Given a path to a folder containing txt. files with trajectory information, extracts this information
    and saves it in a dictionary.
    :param trajectories_path: Path to the folder containing the trajectory files
    :return: A dictionary containing detection information, grouped by (recording_no, frame_no). Individual bees are
    stored in a dictionary containing bee_no, pos_x, pos_y, class_no, angle.
    """

    # Dictionary to store detections grouped by (recording_no, frame_no)
    detections_by_frame = {}

    for root, dirs, files in os.walk(trajectories_path):
        for file in files:
            if file.endswith('.txt'):

                rec_no = root[-1]
                bee_no = file.replace(".txt", "")
                file_path = os.path.join(root, file)
                try:
                    # Open .txt file and extract all information
                    with open(file_path, 'r', encoding='utf-8') as f:
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
                            detections_by_frame[key].append({
                                "bee_no": bee_no,
                                "pos_x": pos_x,
                                "pos_y": pos_y,
                                "class_no": class_no,
                                "angle": angle
                            })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return detections_by_frame


def extract_frames_from_video(videos_path, frames_path):
    """
    Extracts all frames from all videos in the given video_path and saves them to frames_path.
    In case the folder for that recording already exists, that video gets skipped.
    :param videos_path: Path to the folder containing the videos.
    :param frames_path: Path to the folder the frame images are supposed to be saved in.
    """
    for root, dirs, files in os.walk(videos_path):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                rec_no = file.replace(".mp4", "")[-1]

                # If folder already exists, assume that all frames were successfully extracted, might change this later to be more robust
                if os.path.isdir(frames_path + "rec" + rec_no):
                    print(f"Skipping video {file}")
                else:
                    os.makedirs(frames_path + "rec" + rec_no, exist_ok=True)
                    vidcap = cv2.VideoCapture(video_path)
                    count = 0;
                    success = True
                    while success:
                        success, image = vidcap.read()
                        cv2.imwrite(f"{frames_path}/rec{rec_no}/frame%04d.png" % count, image)
                        count += 1


def crop_bee(img, x, y, crop_size=256):
    """
    Crops a square region of size crop_size x crop_size centered at (x, y)
    from a grayscale image. Moves the crop window if necessary to stay within bounds.

    Args:
        img (np.ndarray): The input image.
        x (int): X-coordinate of the center point.
        y (int): Y-coordinate of the center point.
        crop_size (int): Size of the crop (default is 256).

    Returns:
        numpy.ndarray: Cropped image of shape (crop_size, crop_size)
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


def extract_save_crops(bee_detections_by_frame, frames_path,crops_path, bee_number = None):
    """
    Given a dictionary of bee detections, goes through all frames and crops all bees for that frame. If bee_number is specified, only extracts crops for that bee.
    :param bee_detections_by_frame: Dictionary of bee detections
    :param frames_path: Path to folder containing the frames
    :param crops_path: Path to save the cropped images to
    :param bee_number: Optional, to extract crops for only one bee. For testing purposes
    """

    # Make sure that crop dir exists
    os.makedirs(crops_path, exist_ok=True)

    # Loop through all (rec_no, frame_no) pairs and extract all (or the single one) bees for that frame, so frames only have to be opened once each
    for (rec_no, frame_no), bees in sorted(bee_detections_by_frame.items()):
        frame_name = f"frame{str(frame_no).rjust(4, '0')}.png"
        frame_path = os.path.join(frames_path, f"rec{rec_no}", frame_name)
        frame_img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame_img is None:
            print(f"Warning: Could not load frame {frame_path}")
            continue

        for bee in bees:
            #
            if bee["bee_no"] != bee_number:
                continue  # Skip bees that are not the target one

            pos_x = bee["pos_x"]
            pos_y = bee["pos_y"]
            bee_no = bee["bee_no"]
            class_no = bee["class_no"]
            angle = bee["angle"]

            output_name = f"{rec_no}_{frame_no}_{bee_no}_{pos_x}_{pos_y}_{class_no}_{angle}.png"
            output_path = os.path.join(crops_path, output_name)

            # Get crop and save it
            crop = crop_bee(frame_img, pos_x, pos_y, crop_size=256)

            cv2.imwrite(output_path, crop)

def main():
    trajectories_path = "./data/trajectories/"
    vid_path = "./data/videos/"
    frames_path = "./data/frames/"
    crops_path = "./data/crops/"

    bee_detections_by_frame = extract_trajectory_information(trajectories_path)
    extract_frames_from_video(vid_path, frames_path)

    extract_save_crops(bee_detections_by_frame, frames_path,crops_path ,"000000")

    pass

if __name__ == main():
    main()
