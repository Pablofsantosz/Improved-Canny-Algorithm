import cv2
import numpy as np

# Função para criar o ruído artificial (Nível 0.5 = 50% da imagem destruída)
def adicionar_sal_e_pimenta(imagem, intensidade=0.5):
    ruidosa = np.copy(imagem)
    # matrix aleatória de valores entre 0 e 1 com o mesmo tamanho da imagem
    matriz_aleatoria = np.random.rand(*ruidosa.shape)
    
    #divide a intensidade em duas partes: metade para o sal (branco) e metade para a pimenta (preto)
    ruidosa[matriz_aleatoria < (intensidade / 2)] = 0
    
    ruidosa[matriz_aleatoria > (1 - intensidade / 2)] = 255
    return ruidosa



img_limpa = cv2.imread("C:/opencv/opencv/sources/doc/images/lena.png", cv2.IMREAD_GRAYSCALE)  

if img_limpa is None:
    print("Erro: Imagem não encontrada. Coloque a imagem na mesma pasta do script.")
    exit()

print("Pressione QUALQUER TECLA na janela da imagem para avançar os slides do código!")




# Aplicando Sobel (Realce de Bordas dos slides)
sobel_x = cv2.Sobel(img_limpa, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_limpa, cv2.CV_64F, 0, 1, ksize=3)
bordas_sobel = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

# Aplicando Canny Clássico
bordas_canny_limpo = cv2.Canny(img_limpa, 50, 150)

cv2.imshow('1 - Imagem Limpa Original', img_limpa)
cv2.imshow('2 - Sobel (Slide de Aula)', bordas_sobel)
cv2.imshow('3 - Canny Classico (Perfeito)', bordas_canny_limpo)

cv2.waitKey(0) 
cv2.destroyAllWindows()



# O PROBLEMA DO ARTIGO)

# Injetando 50% de ruído sal e pimenta na mesma imagem
img_ruim = adicionar_sal_e_pimenta(img_limpa, intensidade=0.5)

# O Canny Clássico tentando achar as bordas da imagem destruída (ruído 0.5)
bordas_canny_ruim = cv2.Canny(img_ruim, 50, 150)

cv2.imshow('4 - Imagem Destruida (Ruido 0.5)', img_ruim)
cv2.imshow('5 - A Falha do Canny Classico', bordas_canny_ruim)

cv2.waitKey(0)
cv2.destroyAllWindows()