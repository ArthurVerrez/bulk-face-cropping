from os import listdir
from os.path import isfile, join
import argparse
from tqdm import tqdm
import numpy as np
import face_recognition
import imageio.v2 as imageio

IMAGE_EXTENSIONS = ["jpg", "png", "jpeg", "gif", "bmp", ".tif", ".tiff"]


# Simple utility function to crop an image given a bounding box
def crop_image(im, loc):
    (top, right, bottom, left) = loc
    im_crop = np.zeros((bottom - top, right - left, 3))
    for i in range(top, bottom):
        for j in range(left, right):
            im_crop[i - top, j - right] = im[i, j]
    return im_crop.astype(np.uint8)


def extract_faces(input_folder, output_folder, ref_image_path, verbose=False):
    files_to_crop = [
        f
        for f in listdir(input_folder)
        if isfile(join(input_folder, f))
        and f.split(".")[-1].lower() in IMAGE_EXTENSIONS
    ]
    if verbose:
        print(
            f"Found {len(files_to_crop)} picture{'s' if len(files_to_crop)>1 else ''} to crop."
        )

    ref_image = face_recognition.load_image_file(ref_image_path)

    ref_multiple_encodings = face_recognition.face_encodings(ref_image)
    if len(ref_multiple_encodings) == 0:
        raise ValueError(f"No face found in reference image: {ref_image_path}")
    elif len(ref_multiple_encodings) > 1:
        print(
            f"Warning: {len(ref_multiple_encodings)} faces found in reference image: {ref_image_path}. Using the first one."
        )

    ref_encoding = ref_multiple_encodings[0]
    if verbose:
        print(f"Loaded reference image and encoding: {ref_image_path}")

    for f in tqdm(files_to_crop):
        path_to_image = join(input_folder, f)
        save_path = join(
            output_folder, "".join(f.split(".")[:-1]) + "-cropped." + f.split(".")[-1]
        )
        image = face_recognition.load_image_file(path_to_image)

        encodings = face_recognition.face_encodings(image)

        im = imageio.imread(path_to_image)

        face_locations = face_recognition.face_locations(
            image
        )  # (top, right, bottom, left)

        results = face_recognition.compare_faces(encodings, ref_encoding)

        output = None

        for i in range(len(results)):
            if results[i]:
                output = crop_image(im, face_locations[i])
                break

        if output is None:
            print(f"Warning: no face corresponding to the reference found in {f}")
            continue

        imageio.imwrite(save_path, output)

    if verbose:
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from images.")
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        required=True,
        help="Input folder containing the images to crop.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="Output folder where the cropped images will be saved.",
    )
    parser.add_argument(
        "-f",
        "--ref_image_path",
        type=str,
        required=True,
        help="Image containing the face to extract.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Display additional information."
    )
    args = parser.parse_args()

    extract_faces(
        args.input_folder, args.output_folder, args.ref_image_path, args.verbose
    )
