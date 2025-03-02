import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

photo_width = 480
photo_height = 360

window_size = 36
dim_hog_cell = 6

characters_files = {
    "mom": "mom_annotations.txt",
    "dad": "dad_annotations.txt",
    "dexter": "dexter_annotations.txt",
    "deedee": "deedee_annotations.txt",
}


characters_names = {"dad", "mom", "dexter", "deedee", "unknown"}

def show_image(image):
    try:
        if image is None or image.size == 0:
            raise ValueError("The input image is empty or not loaded properly.")
        plt.figure(figsize=(4, 4))
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.show()
    except Exception as e:
        print(f"eroare: {e}")

import matplotlib.pyplot as plt
import cv2

def show_images_side_by_side(image1, image2):
    plt.figure(figsize=(2, 1))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB), cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def cropped_image_of_row(row):
        image_path = os.path.join("..","antrenare", row["Source Directory"], row["Image Name"])
        image = cv.imread(image_path)
        face = image[row["y1"]:row["y2"], row["x1"]:row["x2"]]
        return face

def warp_image_to_ratio(image, target_ratio):

    original_height, original_width = image.shape[:2]
    original_ratio = original_width / original_height

    if target_ratio > original_ratio:
        new_width = int(original_height * target_ratio)
        new_height = original_height
    else:
        new_width = original_width
        new_height = int(original_width / target_ratio)

    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LINEAR)
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_LINEAR)

    cropped_image = resized_image[:new_height, :new_width]

    return cropped_image
