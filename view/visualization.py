# Exemplo de visualização para segmentação
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_image_and_mask(image, mask, pred_mask=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title('Imagem')
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('Máscara Real')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    if pred_mask is not None:
        plt.subplot(1,3,3)
        plt.title('Predição')
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
