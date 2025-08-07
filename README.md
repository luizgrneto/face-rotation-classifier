# Explicação da minha análise

- Ao começar meu estudo do problema, reparei que não havia imagens de exemplos além da imagem já presente no pdf. Cheguei a estraí-la, mas precisaria editar e recortar as diferentes rotações, portanto busquei alguns exemplos na internet. Encontrei um database aberto no Kaggle.

    [Dataset do Kaggle](https://www.kaggle.com/datasets/axondata/selfie-and-official-id-photo-dataset-18k-images)

- Após ponderar o assunto (já que não podemos usar Machine Learning), minha abordagem se focou em analisar a simetria do rosto humano para detectar a rotação da foto. 
    - Fotos em 0 ou 180 graus teriam simetria vertical.
    - Fotos em 90 ou 270 graus teriam simetria horizontal.
    
- Portanto, após uma pesquisa, decidi por utilizar a biblioteca OpenCV, que continha diversas ferramentas que me ajudariam a chegar ao meu objetivo.

- Desenvolvi primeiramente o "split" da imagem entre left, right, top e bottom. Para comparar as respectivas metades, utilizei o método de matchTemplate, que pega uma das metades como template e, sobrepondo com a segunda metade, indica o qual similares elas são dependendo da métrica de análise.

- A métrica de cálculo utilizada foi o TM_CCOEFF_NORMED que faz um cálculo de correlação de intensidade dos pixels, onde +1 é uma correspondência perfeita, e -1 indica que ela é perfeitamente inversa.

- Nesse momento, pela questão do tempo de entrega do challenge e pela falta de fotos de exemplo, decidi **assumir algumas coisas** sobre as fotos do sistema:
    - No dataset que utilizei há fotos de rosto com selfies, o que compromete o check de similaridade proposto acima, dependendo do ângulo do rosto na hora da selfie, fora algumas marcas de flash de câmera em uma foto já impressa, etc.
    - Portanto, assumi que as fotos do sistema seriam no estilo de uma **foto 3x4 de documentos**, onde o rosto está centralizado, a pessoa está evitando usar roupas claras e o fundo é branco. Selecionei do dataset do Kaggle algumas fotos que seguiam essas premissas.

    Com isso, o check de similaridade passou a detectar corretamente nas fotos escolhidas qual era a simetria. Faltava descobrir como detectar a direção do rosto.

- Busquei uma forma de detectar, após a simetria definida, para qual sentido o rosto estava virado. Há algumas tentativas não sucedidas na folder "testes_e_estudos", como tentar fazer uma silhueta preta de todo o formato do corpo a pessoa na foto, de detectar o contorno do rosto, detectar olhos usando blob detection, usar Sobel para detectar bordas como boca, olhos, etc. 

- Portanto, como assumi acima que as fotos seriam no formato 3x4 de documento, com fundo branco, implementei uma classificação de de sentido do rosto que calcula a média de intensidade de pixels nas metades não relacionadas à simetria da foto.

    Ex: Uma foto com simetria vertical (0 ou 180 graus) vai estar rotacionada em 180 graus se o "bottom" tiver maior intensidade média de pixels que o "top", porque o fundo branco ocupa mais espaço acima da cabeça da pessoa nesse tipo de foto e isso eleva a média de intensidade.

- Com isso, nas imagens que utilizei de exemplo, a classificação final de rotação foi bem sucedida.

- Ao fim, eu salvo o resultado da classificação em um json com o nome da imagem, contendo tanto o path quanto o resultado da classificação. O formato JSON seria um "coringa" caso seja necessário levar a classificação para um banco de dados ou exibí-la em outro lugar.

- Acrescentei ao fim um blur Gaussiano para suaviação, pois no início da análise das fotos do Kaggle havia algumas com muito ruído. A idéia era criar uma classe que possibilitasse habilitar/desabilitar essas ferramentas, utilizar outro filtro, escolher o kernel, etc. Mas por falta de tempo, acabei deixando só alguns parâmetros da Gaussiana mesmo.

- **Pontos de melhoria**
    - Não tive tempo de testar outros coeficientes fora o TM_CCOEFF_NORMED que já era o default no código que pesquisei. Talvez um outro coeficiente melhorasse a performance do modelo, ou ainda forneceria algum insight para detectar a direção do rosto após a simetria.
    - Não havia nenhum dataset com exemplos de foto e eu levei algum tempo para achar exemplos na internet (até encontrar o Kaggle), portanto não consegui pensar, em virtude do tempo, em nenhuma métrica referente à classificação para embasar melhor esse estudo.
    - Porém, entendo como válido para demonstrar meu raciocínio de desenvolvimento, já que não utilizar Machine Learning trouxe um novo grau de desafio (muito bem vindo) à esse case.

Abaixo segue o Readme-md que eu criaria para um repositório como esse.

---

# FaceRotationClassifier

Classificador de rotação de rosto em imagens usando análise de simetria.  
Utiliza **OpenCV** e **NumPy** para processar imagens em tons de cinza, detectar simetria e estimar a rotação do rosto (0°, 90°, 180° ou 270°).  
A saída inclui um **arquivo JSON** com o nome da imagem e o ângulo estimado.


## Funcionalidades

- Leitura da imagem em tons de cinza
- Aplicação opcional de **filtro Gaussiano** para suavização
- Análise de **simetria esquerda-direita e cima-baixo** usando template matching
- **Estimativa da rotação do rosto** em graus
- **Geração automática de JSON** com o resultado, usando o mesmo nome da imagem


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
│   └── outputs/                # Saídas (JSONs)
│
├── testes_e_estudos/           # Testes e estudos
│   └── testes_e_estudos.ipynb  # Notebook com alguns estudos e testes que não avançaram 
│
├── README.md
├── requirements.txt            # dependências
└── Tutorial.ipynb              # Notebook tutorial para usar o classificador
```


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
