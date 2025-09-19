import os
import numpy as np
from PIL import Image

def calc_holo_moyen(path, cam_nb_pix_X, cam_nb_pix_Y, type_image):
    images = [os.path.join(path, img) for img in os.listdir(path) if img.lower().endswith(type_image.lower())]
    holo_sum = np.zeros((cam_nb_pix_Y, cam_nb_pix_X), dtype=np.float32)
    for image_path in images:
        holo = read_image(image_path, cam_nb_pix_X, cam_nb_pix_Y)
        holo_sum += holo
    return holo_sum / len(images)

def read_image(image_path, cam_nb_pix_X, cam_nb_pix_Y):
    img = Image.open(image_path)
    return np.array(img)

def display(image):
    img = Image.fromarray(image)
    img.show()
