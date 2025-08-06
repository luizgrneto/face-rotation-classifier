# FaceRotationClassifier

Classificador de rotação de rosto em imagens usando análise de simetria.  
Utiliza **OpenCV** e **NumPy** para processar imagens em tons de cinza, detectar simetria e estimar a rotação do rosto (0°, 90°, 180° ou 270°).  
A saída inclui um **arquivo JSON** com o nome da imagem e o ângulo estimado.

---

## Funcionalidades

- Leitura da imagem em tons de cinza
- Aplicação opcional de **filtro Gaussiano** para suavização
- Análise de **simetria esquerda-direita e cima-baixo** usando template matching
- **Estimativa da rotação do rosto** em graus
- **Geração automática de JSON** com o resultado, usando o mesmo nome da imagem

---

## Instalação

Este projeto depende das bibliotecas:

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Você pode instalar as dependências com:

```bash
pip install opencv-python numpy matplotlib
```

ou

```bash
pip install -r requirements.txt
```

---

## Estrutura do Projeto

```
FaceRotationClassifier/
├── face_rotation/
│   ├── __init__.py
│   └── classifier.py           # Contém a classe FaceRotationClassifier
│
├── data/
│   ├── docs/                   # Documentos (PDF do teste técnico)
│   ├── images/                 # Imagens de entrada para teste
│   └── outputs/                # Saídas (JSONs, logs, etc.)
│
├── testes_e_estudos/           # (opcional) testes unitários
│   └── testes_e_estudos.ipynb  # Notebook com alguns estudos e testes que não avançaram 
│
├── README.md
├── requirements.txt            # dependências
└── Tutorial.ipynb              # Notebook tutorial para usar o classificador
```

---

## Exemplo de Uso

```python
from face_rotation.classifier import FaceRotationClassifier

classifier = FaceRotationClassifier(
    image_path="data/images/African_female_32_Passport.jpg",
    output_path="data/outputs"  # diretório onde o JSON será salvo
)

classifier.run(show_image=False)
```

### Saída esperada no terminal:

```
A imagem está rotacionada em 0 graus.
Resultado salvo em data/outputs/African_female_32_Passport.json
```

### Exemplo de JSON gerado:

```json
{
    "image_path": "data/images/African_female_32_Passport.jpg",
    "rotation_degrees": 90
}
```
