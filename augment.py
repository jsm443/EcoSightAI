import os
from PIL import Image
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def augment(folder, target_num):
    """augment uses rotation and flipping to generate new images for training data.
    the new images are saved in folder

    parameters:
    folder: the path of the folder that the augmentation should be performed on
    target_num: the number of total images desired within that folder


    the folder is required to contain fewer images than target_num
    """

    # Check if the folder exists
    if not os.path.exists(folder):
        raise ValueError(f"The folder {folder} does not exist.")

    # Get the list of image files in the folder
    image_files = [
        f for f in os.listdir(folder) if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    num_existing_images = len(image_files)

    if num_existing_images >= target_num:
        raise ValueError(
            f"The folder already contains {num_existing_images} images, which is greater than or equal to the target number {target_num}."
        )

    # Initialize the ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=40, horizontal_flip=True, vertical_flip=True
    )

    # Calculate how many new images are needed
    num_new_images_needed = target_num - num_existing_images

    # Perform augmentation
    generated_count = 0
    while generated_count < num_new_images_needed:
        for image_file in image_files:
            if generated_count >= num_new_images_needed:
                break

            # Load the image
            image_path = os.path.join(folder, image_file)
            image = Image.open(image_path)
            image_array = np.array(image)
            image_array = image_array.reshape(
                (1,) + image_array.shape
            )  # Reshape for the generator

            # Generate new images
            for batch in datagen.flow(
                image_array,
                batch_size=1,
                save_to_dir=folder,
                save_prefix="aug",
                save_format="jpeg",
            ):
                generated_count += 1
                if generated_count >= num_new_images_needed:
                    break

    print(f"Augmentation complete. {generated_count} new images generated in {folder}.")
