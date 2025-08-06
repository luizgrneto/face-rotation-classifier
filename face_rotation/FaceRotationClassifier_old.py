from cv2 import imread, GaussianBlur, matchTemplate, IMREAD_GRAYSCALE, TM_CCOEFF_NORMED
from numpy import fliplr, flipud, mean
from typing import Tuple, Dict
from numpy import ndarray
import matplotlib.pyplot as plt

class FaceRotationClassifier:

    def __init__(self, image_path: str, imread_mode: int = IMREAD_GRAYSCALE, apply_gaussian: bool = True, kernel_size: Tuple = (5,5)):
        
        self.image_path = image_path
        self.imread_mode = imread_mode
        self.apply_gaussian = apply_gaussian
        self.kernel_size = kernel_size

    def open_image(self, image_path: str, imread_mode: int) -> ndarray:
        return imread(filename= image_path, 
                      flags = imread_mode)

    def plot_img(self, image):
        plt.imshow(image, cmap='gray') 
        plt.axis('off')
        plt.show()

    def apply_gaussian_blur(self, image: ndarray, kernel_size: Tuple) -> ndarray:
        return GaussianBlur(src=image, 
                            ksize=kernel_size, 
                            sigmaX=0, 
                            sigmaY=0)

    def split_image(self, gray_image: ndarray) -> Tuple:

        h, w = gray_image.shape

        mid_w = w // 2
        left = gray_image[:, :mid_w]
        right = fliplr(gray_image[:, -mid_w:]) 

        mid_h = h // 2
        top = gray_image[:mid_h, :]
        bottom = flipud(gray_image[-mid_h:, :]) 

        return (left, right), (top, bottom)

    def check_symmetry(self, image_parts: Tuple) -> Dict:
        return {
            "left-right symmetry": matchTemplate(image_parts[0][0], image_parts[0][1], TM_CCOEFF_NORMED)[0,0],
            "top-bottom symmetry": matchTemplate(image_parts[1][0], image_parts[1][1], TM_CCOEFF_NORMED)[0,0]
            }
    
    def detect_face_rotation(self, image_parts: Tuple, symmetry_result: Dict) -> int:

        left_mean = mean(image_parts[0][0])
        right_mean = mean(image_parts[0][1])
        top_mean = mean(image_parts[1][0])
        bottom_mean = mean(image_parts[1][1])

        if symmetry_result["left-right symmetry"] > symmetry_result["top-bottom symmetry"]:
            if top_mean > bottom_mean:
                degrees = 0
            else:
                degrees = 180
        else:
            if left_mean > right_mean:
                degrees = 90
            else:
                degrees = 270

        return degrees
    
    def run(self):

        gray_img = self.open_image(image_path = self.image_path, 
                              imread_mode = self.imread_mode)
        
        if self.apply_gaussian:
            gray_img = self.apply_gaussian_blur(image= gray_img, 
                                                kernel_size=self.kernel_size
                                                )
            
        self.plot_img(gray_img)

        img_parts = self.split_image(gray_image=gray_img)

        symmetry_check_dict = self.check_symmetry(image_parts=img_parts)

        degrees = self.detect_face_rotation(image_parts=img_parts, 
                                            symmetry_result=symmetry_check_dict
                                            )
        print(f"Image has {degrees} degrees rotation.")


