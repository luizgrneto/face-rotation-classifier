import json
import os

from cv2 import imread, GaussianBlur, matchTemplate, IMREAD_GRAYSCALE, TM_CCOEFF_NORMED
from numpy import fliplr, flipud, mean, ndarray
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


class FaceRotationClassifier:
    """
    Classifica a rotação de um rosto em uma imagem a partir de uma
    estimativa em cima da análise de simetria da imagem e intensidade
    média.
    """

    def __init__(self, image_path: str, output_path: Optional[str] = None, 
                 imread_mode: int = IMREAD_GRAYSCALE,
                 apply_gaussian: bool = True, kernel_size: Tuple[int, int] = (5, 5)):
        """
        Inicializa o classificador.

        Args:
            image_path (str): Caminho da imagem a ser processada.
            output_path (str, optional): Caminho do arquivo JSON de saída. (padrão: None)
            imread_mode (int, optional): Flag do OpenCV para leitura da imagem (padrão: IMREAD_GRAYSCALE).
            apply_gaussian (bool, optional): Se True, aplica blur Gaussiano na imagem (padrão: True).
            kernel_size (tuple[int, int], optional): Tamanho do kernel para o blur Gaussiano (padrão: (5,5)).
        """
        self.image_path = image_path
        self.output_path = output_path
        self.imread_mode = imread_mode
        self.apply_gaussian = apply_gaussian
        self.kernel_size = kernel_size

    def open_image(self, image_path: str, imread_mode: int) -> ndarray:
        """
        Lê a imagem do disco com o modo de leitura escolhido. Por padrão,
        o modo de leitura é em escalas de cinza.

        Args:
            image_path (str): Caminho da imagem.
            imread_mode (int): Flag do OpenCV para leitura da imagem.

        Returns:
            numpy.ndarray: Matriz da imagem carregada.

        Raises:
            FileNotFoundError: Se a imagem não for encontrada ou não puder ser carregada.
        """
        img = imread(filename=image_path, flags=imread_mode)
        if img is None:
            raise FileNotFoundError(f"Não foi possível ler a imagem em '{image_path}'")
        return img

    def plot_img(self, image: ndarray):
        """
        Exibe a imagem em escala de cinza usando matplotlib.

        Args:
            image (numpy.ndarray): Imagem a ser exibida.
        """
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    def apply_gaussian_blur(self, image: ndarray, kernel_size: Tuple[int, int]) -> ndarray:
        """
        Aplica blur Gaussiano na imagem.

        Args:
            image (numpy.ndarray): Imagem de entrada.
            kernel_size (tuple[int, int]): Tamanho do kernel para o blur.

        Returns:
            numpy.ndarray: Imagem suavizada.
        """
        return GaussianBlur(src=image, ksize=kernel_size, sigmaX=0, sigmaY=0)

    def split_image(self, gray_image: ndarray) -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]:
        """
        Divide a imagem em partes para análise de simetria:
        - Esquerda e direita (com flip horizontal)
        - Superior e inferior (com flip vertical)

        Args:
            gray_image (numpy.ndarray): Imagem em tons de cinza.

        Returns:
            tuple: ((left, right), (top, bottom)) para análise de simetria.
        """
        h, w = gray_image.shape

        # Simetria vertical
        mid_w = w // 2
        left = gray_image[:, :mid_w]
        right = fliplr(gray_image[:, -mid_w:])

        # Simetria horizontal
        mid_h = h // 2
        top = gray_image[:mid_h, :]
        bottom = flipud(gray_image[-mid_h:, :])

        return (left, right), (top, bottom)

    def check_symmetry(self, image_parts: Tuple) -> Dict[str, float]:
        """
        Calcula scores de simetria da imagem com correlação cruzada normalizada.

        Args:
            image_parts (tuple): Tupla no formato ((left, right), (top, bottom)).

        Returns:
            dict: Scores de simetria esquerda-direita e cima-baixo.
        """
        return {
            "left-right symmetry": matchTemplate(image_parts[0][0], image_parts[0][1], TM_CCOEFF_NORMED)[0, 0],
            "top-bottom symmetry": matchTemplate(image_parts[1][0], image_parts[1][1], TM_CCOEFF_NORMED)[0, 0]
        }

    def detect_face_rotation(self, image_parts: Tuple, symmetry_result: Dict[str, float]) -> int:
        """
        Estima a rotação do rosto baseado no score de simetria e na intensidade média.
        Exemplo: uma simetria horizontal (left-right) com intensidade média maior na 
        parte superior (top), indica que a rotação possivelmente é de zero graus já que
        nas fotos de identidade há mais pixels de fundo branco nessa região.

        Args:
            image_parts (tuple): Partes da imagem para análise de simetria.
            symmetry_result (dict): Resultado do cálculo de simetria.

        Returns:
            int: Ângulo estimado em graus (0, 90, 180 ou 270).
        """
        left_mean = mean(image_parts[0][0])
        right_mean = mean(image_parts[0][1])
        top_mean = mean(image_parts[1][0])
        bottom_mean = mean(image_parts[1][1])

        if symmetry_result["left-right symmetry"] > symmetry_result["top-bottom symmetry"]:
            degrees = 0 if top_mean > bottom_mean else 180
        else:
            degrees = 90 if left_mean > right_mean else 270

        return degrees

    def run(self, show_image: bool = True) -> int:
        """
        Executa o pipeline completo:
        1. Lê a imagem
        2. Aplica pré-processamento (blur opcional)
        3. Calcula score de simetria
        4. Estima a rotação do rosto pela média de intensidade dos pixels nas 
           outras metades da imagem.
        5. Salva um JSON com o resultado da classificação e o path da imagem. Se 
           não for definido um output_path, ele salvará no mesmom path da imagem.

        Args:
            show_image (bool, optional): Se True, exibe a imagem processada (padrão: True).

        Returns:
            int: Ângulo estimado em graus.
        """
        gray_img = self.open_image(image_path=self.image_path, imread_mode=self.imread_mode)

        if self.apply_gaussian:
            gray_img = self.apply_gaussian_blur(image=gray_img, kernel_size=self.kernel_size)

        if show_image:
            self.plot_img(gray_img)

        img_parts = self.split_image(gray_image=gray_img)
        symmetry_check_dict = self.check_symmetry(image_parts=img_parts)
        degrees = self.detect_face_rotation(image_parts=img_parts, symmetry_result=symmetry_check_dict)

        print(f"A imagem está rotacionada em {degrees} graus.")

        # Montar caminho final do JSON
        img_name = os.path.splitext(os.path.basename(self.image_path))[0] + ".json"
        if self.output_path:
            json_path = os.path.join(self.output_path, img_name)
        else:
            json_path = os.path.join(os.path.dirname(self.image_path), img_name)

        # Criar diretório, se necessário
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Salvar JSON
        result = {
            "image_path": self.image_path,
            "rotation_degrees": degrees
        }
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

        print(f"Resultado salvo em {json_path}")

