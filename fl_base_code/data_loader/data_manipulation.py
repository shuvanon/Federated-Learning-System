import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image


class DataManipulation:
    """
    DataManipulation class for applying various image manipulation techniques.

    Args:
        manipulation_config (dict): Configuration for the image manipulation techniques.

    Returns:
        None
    """

    def __init__(self, manipulation_technique: str, manipulation_config: dict) -> None:
        self.manipulation_technique = manipulation_technique
        self.manipulation_config = manipulation_config
        self.random_mode = self.manipulation_config['manipulation_mode']

        if self.random_mode == 'random':
            self.manipulation_value = self.random_value()
        else:
            self.manipulation_value = self.fixed_value()
        # self.manipulation_value = manipulation_value

    def random_value(self):
        if self.manipulation_technique == "contrast":
            manipulation_value = random.uniform(self.manipulation_config[self.manipulation_technique]["alpha_min"],
                                                self.manipulation_config[self.manipulation_technique]["alpha_max"])

        elif self.manipulation_technique == "brightness":
            manipulation_value = random.randint(self.manipulation_config[self.manipulation_technique]["intensity_min"],
                                                self.manipulation_config[self.manipulation_technique]["intensity_max"])

        elif self.manipulation_technique == "white_balance":
            manipulation_value = random.randint(
                self.manipulation_config[self.manipulation_technique]["color_temperature_min"],
                self.manipulation_config[self.manipulation_technique]["color_temperature_max"])

        elif self.manipulation_technique == "gaussian_noise":
            manipulation_value = random.randint(self.manipulation_config[self.manipulation_technique]["sigma_min"],
                                                self.manipulation_config[self.manipulation_technique]["sigma_max"])

        elif self.manipulation_technique == "edge_filter":
            lower = random.uniform(self.manipulation_config[self.manipulation_technique]["lower_threshold"],
                                   self.manipulation_config[self.manipulation_technique]["upper_threshold"])
            high = random.uniform(lower, self.manipulation_config[self.manipulation_technique]["upper_threshold"])

        elif self.manipulation_technique == "sharpening_blurring":
            manipulation_value = random.uniform(self.manipulation_config[self.manipulation_technique]["lambda_min"],
                                                self.manipulation_config[self.manipulation_technique]["lambda_max"])
        else:
            raise ValueError(f"Technique '{self.manipulation_technique}' is not supported or implemented.")
        return manipulation_value

    def fixed_value(self):
        if self.manipulation_technique == "contrast":
            manipulation_value = self.manipulation_config[self.manipulation_technique]["alpha"]

        elif self.manipulation_technique == "brightness":
            manipulation_value = self.manipulation_config[self.manipulation_technique]["intensity"]

        elif self.manipulation_technique == "white_balance":
            manipulation_value = self.manipulation_config[self.manipulation_technique]["color_temperature"]

        elif self.manipulation_technique == "gaussian_noise":
            manipulation_value = self.manipulation_config[self.manipulation_technique]["sigma"]

        elif self.manipulation_technique == "edge_filter":
            lower = self.manipulation_config[self.manipulation_technique]["lower_threshold"]
            high = self.manipulation_config[self.manipulation_technique]["upper_threshold"]

        elif self.manipulation_technique == "sharpening_blurring":
            manipulation_value = self.manipulation_config[self.manipulation_technique]["lambda_value"]
        else:
            raise ValueError(f"Technique '{self.manipulation_technique}' is not supported or implemented.")
        return manipulation_value

    def bulk_process(self, client_id: int, img_dir: str, output_dir: str) -> None:
        """
        Processes and saves the manipulated images for each client's data.

        Args:
            client_id (int): ID of the client whose data is being processed.
            img_dir (str): Directory containing the original images.
            manipulation_config (dict): Configuration for the manipulation technique.
            output_dir (str): Directory to save the manipulated images.

        Returns:
            None
        """
        print(
            f"Processing client {client_id}... with Manipulation technique: {self.manipulation_technique} and Manipulation value: {self.manipulation_value}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data_frame = pd.read_csv(f'{output_dir}/data.csv')

        for _, row in data_frame.iterrows():
            img_path = os.path.join(img_dir, str(row.iloc[0]) + ".png")
            # print("IMAGE PATH:  "+img_path)
            img = Image.open(img_path)

            if self.manipulation_technique == "contrast":
                manipulated_img = self.enhanced_contrast(img, self.manipulation_value)
            elif self.manipulation_technique == "brightness":
                manipulated_img = self.adjust_brightness(img, self.manipulation_value)
            elif self.manipulation_technique == "white_balance":
                manipulated_img = self.adjust_white_balance(img, self.manipulation_value)
            elif self.manipulation_technique == "gaussian_noise":
                manipulated_img = self.add_gaussian_noise(img, self.manipulation_value)
            elif self.manipulation_technique == "edge_filter":
                manipulated_img = self.apply_edge_filter(img, self.manipulation_value, self.manipulation_value)
            elif self.manipulation_technique == "sharpening_blurring":
                manipulated_img = self.apply_sharpening_or_blurring(img, self.manipulation_value)
            else:
                raise ValueError(f"Technique '{self.manipulation_technique}' is not supported or implemented.")

            save_path = os.path.join(output_dir, str(row.iloc[0]) + ".png")
            # print("SAVE PATH:  "+save_path)
            manipulated_img.save(save_path)

        # Save manipulated data
        manipulated_data_csv = os.path.join(output_dir, 'data.csv')
        data_frame.to_csv(manipulated_data_csv, index=False)  # Ensure the CSV is saved here
        print(f"Manipulated data saved at {manipulated_data_csv}")

    def process_in_memory(self, data_frame: pd.DataFrame, img_dir: str) -> pd.DataFrame:
        """
        Processes and returns the manipulated images in memory for each client's data.

        Args:
            data_frame (pd.DataFrame): DataFrame containing image filenames and labels.
            img_dir (str): Directory containing the original images.

        Returns:
            pd.DataFrame: DataFrame with manipulated images in memory.
        """
        manipulated_data = []

        for _, row in data_frame.iterrows():
            img_path = os.path.join(img_dir, row[0])
            img = Image.open(img_path)

            if self.manipulation_config['type'] == 'enhanced_contrast':
                img = self.enhanced_contrast(img, self.manipulation_config['factor'])

            manipulated_data.append((img, row[1]))  # Append image and label

        return pd.DataFrame(manipulated_data, columns=['image', 'label'])

    def apply_randomized_manipulation(self, img: Image.Image) -> Image.Image:
        """
        Applies the specified manipulation technique with a randomly chosen parameter value.

        Args:
            img (Image.Image): The image to be processed.

        Returns:
            Image.Image: The manipulated image.
        """
        if self.manipulation_technique == "contrast":
            alpha = random.uniform(self.manipulation_config[self.manipulation_technique]["alpha_min"],
                                   self.manipulation_config[self.manipulation_technique]["alpha_max"])
            return self.enhanced_contrast(img, alpha)

        elif self.manipulation_technique == "brightness":
            intensity = random.uniform(self.manipulation_config[self.manipulation_technique]["intensity_min"],
                                       self.manipulation_config[self.manipulation_technique]["intensity_max"])
            return self.adjust_brightness(img, intensity)

        elif self.manipulation_technique == "white_balance":
            color_temperature = random.uniform(
                self.manipulation_config[self.manipulation_technique]["color_temperature_min"],
                self.manipulation_config[self.manipulation_technique]["color_temperature_max"])
            return self.adjust_white_balance(img, color_temperature)

        elif self.manipulation_technique == "gaussian_noise":
            sigma = random.uniform(self.manipulation_config[self.manipulation_technique]["sigma_min"],
                                   self.manipulation_config[self.manipulation_technique]["sigma_max"])
            return self.add_gaussian_noise(img, sigma)

        elif self.manipulation_technique == "edge_filter":
            lower = random.uniform(self.manipulation_config[self.manipulation_technique]["lower_threshold"],
                                   self.manipulation_config[self.manipulation_technique]["upper_threshold"])
            high = random.uniform(lower, self.manipulation_config[self.manipulation_technique]["upper_threshold"])
            return self.apply_edge_filter(img, lower, high)

        elif self.manipulation_technique == "sharpening_blurring":
            lambda_value = random.uniform(self.manipulation_config[self.manipulation_technique]["lambda_min"],
                                          self.manipulation_config[self.manipulation_technique]["lambda_max"])
            return self.apply_sharpening_or_blurring(img, lambda_value)

        else:
            raise ValueError(f"Technique '{self.manipulation_technique}' is not supported or implemented.")

    def enhanced_contrast(self, img: Image.Image, alpha) -> Image.Image:

        img_array = np.array(img, dtype='float32')
        img_array = np.clip(alpha * img_array, 0, 255)
        adjusted_img = Image.fromarray(img_array.astype('uint8'))
        return adjusted_img

    def adjust_brightness(self, img: Image.Image, intensity) -> Image.Image:

        img_array = np.array(img, dtype='float32')
        img_array = np.clip(img_array + intensity, 0, 255)
        adjusted_img = Image.fromarray(img_array.astype('uint8'))
        return adjusted_img

    def adjust_white_balance(self, img: Image.Image, color_temperature) -> Image.Image:
        def kelvin_to_rgb(temp_k):
            temp_k = temp_k / 100
            if temp_k <= 66:
                r = 255
            else:
                r = temp_k - 60
                r = 329.698727446 * (r ** -0.1332047592)
                r = max(0, min(255, r))

            if temp_k <= 66:
                g = temp_k
                g = 99.4708025861 * np.log(g) - 161.1195681661
                g = max(0, min(255, g))
            else:
                g = temp_k - 60
                g = 288.1221695283 * (g ** -0.0755148492)
                g = max(0, min(255, g))

            if temp_k >= 66:
                b = 255
            elif temp_k <= 19:
                b = 0
            else:
                b = temp_k - 10
                b = 138.5177312231 * np.log(b) - 305.0447927307
                b = max(0, min(255, b))

            return r / 255, g / 255, b / 255

        r_scale, g_scale, b_scale = kelvin_to_rgb(color_temperature)
        img_array = np.array(img, dtype='float32')
        img_array[..., 0] *= r_scale
        img_array[..., 1] *= g_scale
        img_array[..., 2] *= b_scale
        img_array = np.clip(img_array, 0, 255)
        adjusted_img = Image.fromarray(img_array.astype('uint8'))
        return adjusted_img

    def apply_edge_filter(self, img: Image.Image, lower_threshold, upper_threshold) -> Image.Image:

        img_gray = np.array(img.convert('L'))
        edges = cv2.Canny(img_gray, lower_threshold, upper_threshold)
        edge_img = Image.fromarray(edges)
        return edge_img

    def add_gaussian_noise(self, img: Image.Image, sigma) -> Image.Image:

        if sigma < 0:
            raise ValueError("Standard deviation (sigma) must be non-negative.")

        img_array = np.array(img, dtype='float32')
        noise = np.random.normal(0, sigma, img_array.shape)
        noisy_img_array = img_array + noise
        noisy_img_array = np.clip(noisy_img_array, 0, 255)
        noisy_img = Image.fromarray(noisy_img_array.astype('uint8'))
        return noisy_img

    def apply_sharpening_or_blurring(self, img: Image.Image, lambda_value) -> Image.Image:

        img_array = np.array(img, dtype='float32')
        blurred_img_array = cv2.GaussianBlur(img_array, (5, 5), sigmaX=0)
        mask = img_array - blurred_img_array
        output_img_array = img_array + lambda_value * mask
        output_img_array = np.clip(output_img_array, 0, 255)
        output_img = Image.fromarray(output_img_array.astype('uint8'))
        return output_img
