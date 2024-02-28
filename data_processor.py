import numpy as np
from PIL import Image
import os


class DataProcessor:
    img_rows, img_cols = 48, 48

    @classmethod
    def read_images(cls, path):
        image_folders = os.listdir(path)
        images = []
        labels = []

        for folder in image_folders:
            for name in os.listdir(os.path.join(path, folder)):
                image_path = os.path.join(path, folder, name)

                with Image.open(image_path) as image:
                    image = image.convert('L')
                    image = image.resize((cls.img_rows, cls.img_cols))
                    image = np.array(image, dtype=np.float32)
                    image /= 255
                    images.append(image)
                split_path = image_path.split("\\")
                text_label = split_path[len(split_path) - 2]
                label = []
                if text_label == 'angry':
                    label = [1, 0, 0, 0, 0, 0]
                elif text_label == 'neutral':
                    label = [0, 1, 0, 0, 0, 0]
                elif text_label == 'fearful':
                    label = [0, 0, 1, 0, 0, 0]
                elif text_label == 'happy':
                    label = [0, 0, 0, 1, 0, 0]
                elif text_label == 'sad':
                    label = [0, 0, 0, 0, 1, 0]
                elif text_label == 'surprised':
                    label = [0, 0, 0, 0, 0, 1]

                labels.append(label)
                print(f'{image_path} read')
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    @classmethod
    def read_image(cls, path):
        images = []

        with Image.open(path) as image:
            image = image.convert('L')
            image = image.resize((cls.img_rows, cls.img_cols))
            image = np.array(image, dtype=np.float32)
            image /= 255
            images.append(image)

            print(f'{path} read')
        images = np.array(images)
        return images
