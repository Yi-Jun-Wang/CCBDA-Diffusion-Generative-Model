# Diffusion Generative Model

Implementing diffusion generative model using MNIST dataset.

## Dataset
- Unzip mnist.zip to `./datas`
    ```sh
    unzip mnist.zip -d ./datas
    ```
- Folder structure
    ```
    ./
    │── datas/
    ├── Readme.md
    ├── requirements.txt
    ├── train.py
    ├── model.py
    |── dataset.py
    |── generate_image.py
    ```

## Environment
- Python 3.8
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python train.py --epochs 50
```

## generate_images & Diffusion Process
```sh
python generate_image.py 
```
