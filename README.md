# Dog Breed Recognition :dog: <br><br>

## 1. A ideia:

Desenvolver um sistema de reconhecimento de raças de cachorro. Porém, com uma dificuldade extra: adicionar raças que não foram vistas em tempo de treinamento, e saber identificar *unknowns.*


### O problema foi dividido em 3 partes + Aplicação Web: <br>
- **Primeira parte (Treinamento):**
    - Treinar um modelo com um total de **100** raças de cachorros;
    - Apresentar os resultados desse modelo; <br><br>

- **Segunda parte (Enroll):**
    - Criar um sistema para lidar com raças ainda não vistas;
    - Serão 20 novas raças e o sistema deverá reconhecer somente essas;
    - O processo de enroll deverá ser rápido (menos de 1 segundo por imagem); <br><br>

- **Terceira parte (Unknown):**
    - Treinar um modelo com um total de **100** raças de cachorros;
    - Apresentar os resultados desse modelo; 

- **Aplicação Web:**
    - Aplicação web com front-end simples;
    - Deverá ser possível fazer o upload da imagem e verificar o reconhecimento do sistema;
    - Deverá haver uma maneira fácil de executar a aplicação;

<br>

## 2. Instalação <br>
- Este projeto foi desenvolvido utilizando Python na versão [`3.7.1`](https://www.python.org/downloads/release/python-371/) e foi executado nos sistemas operacionais `Windows 10` (recomendado) e `Ubuntu 20.04` (*detalhe que ao rodar no ubuntu percebi problemas ao carregar os pesos treinados no windows*).

- O primeiro passo é instalar as dependências do ambiente presente no arquivo `requirements.txt` na raiz do projeto;
    - ```pip install -r requirements.txt```

- É importante deixar claro que o **dataset** utilizado nesse problema é de posse exclusiva da empresa que ofertou esse desafio, por tanto, ele só será disponibilizado caso eu seja autorizado a fazer o mesmo;

    - O dataset deverá ser extraído para a pasta `/data` conforme apresentado na sessão \#4 (Estrutura);

- Caso tenha instalado os `requirements.txt` em um ambiente virtual, talvez seja necessário registrar esse ambiente no `jupyter notebook`. Pode-se fazer isso executando os seguintes passos:
    - `conda install -c anaconda ipykernel`
    - `python -m ipykernel install --user --name={nome-do-env}`

- É **necessário** fazer [download](https://drive.google.com/file/d/1lmXQy3a4nZ1b3BDPZ2lE_TKpPdFosFuM/view?usp=sharing) dos dados pertencentes a pasta `/data`;

- É **necessário** fazer [download](https://drive.google.com/file/d/1nXhO9oe2rH3wlqgdTDbL2PBfk3AnkaPd/view?usp=sharing) dos pesos dos modelos pertencentes a pasta `/models`;


<br>

## 3. Executando os códigos
Não existe uma ordem definada para executar os arquivos deste repositório, mas para definir uma ordem de resolução do problema dado o desafio proposto devem ser executados os seguintes arquivos:

*É importante mencionar que caso haja interesse em treinar os modelos novamente, será requisitado um cadastro na ferramenta [wandb](https://wandb.ai/).*

### Primeira Parte (Treinamento)
- /notebooks/Dog-Breed-Data-Analysis.ipynb (análise do dataset)
- /notebooks/Dog-Breed-Classification.ipynb (treinamento de modelo - pré-treino - ResNet18)
- /notebooks/Dog-Breed-Classification-Baseline.ipynb (treinamento de modelo - baseline - ResNet18)
- /notebooks/Dog-Breed-Classification-Baseline.ipynb (treinamento de modelo - ResNet50)
- /notebooks/Dog-Breed-Results.ipynb (análise dos resultados)

### Segunda e Terceira Parte (Enroll + Unknown)
- /notebooks/Dog-Breed-Feature-Extractor.ipynb (extração de features)
- /notebooks/Dog-Breed-Enroll-Unknown.ipynb (criação de modelo enroll + unknown)

### Aplicação Web
- /src/app.py (Aplicação Web utilizando [Streamlit.io](https://streamlit.io/))
- para acessar a aplicação basta executar o comando ```streamlit run app.py```


## 4. Estrutura

    - /data 
        - enroll_dict.pkl
        - resnet50-test_enroll_features_labels.pickle
        - resnet50-train_enroll_features.pickle
        - resnet50-validation_features_labels.pickle
        - test_enroll_features_labels.pickle
        - train_enroll_features.pickle
        - validation_features_labels.pickle
        
        - /dogs
            - /train
            - /recognition
                - /enroll
                - /test

    - /models
        - /knn-breed 
        - /knn-unknown
        - /resnet-18-baseline
        - /resnet-18-enhanced-fc
        - /resnet-18-enhanced-layer4
        - /resnet-50

    - /notebooks
        - Dog-Breed-Classification-Baseline.ipynb
        - Dog-Breed-Classification-ResNet50.ipynb
        - Dog-Breed-Classification.ipynb
        - Dog-Breed-Data-Analysis.ipynb
        - Dog-Breed-Enroll-Unknown.ipynb
        - Dog-Breed-Feature-Extractor.ipynb
        - Dog-Breed-Results.ipynb

    - /reports
        - /figures
        - /results

    - /src
        - /app.py
        - /dataset.py
        - /model.py

    - README.md
    - requirements.txt