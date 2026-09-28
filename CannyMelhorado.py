import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_noise(image):
    """
    Detecção aprimorada de ruído impulsivo (Sal e Pimenta).
    Verifica extremos locais e discrepância estatística em relação aos vizinhos.
    """
    h, w = image.shape
    noise_map = np.zeros((h, w), dtype=np.uint8)
    pad_img = np.pad(image, 1, mode='reflect')

    for i in range(1, h + 1):
        for j in range(1, w + 1):
            window = pad_img[i-1:i+2, j-1:j+2]
            center = pad_img[i, j]
            
            # Pixels saturados em 0 ou 255 são candidatos imediatos a sal e pimenta
            if center == 0 or center == 255:
                # Se nem todos os vizinhos forem iguais, é ruído isolado
                if not (np.all(window == 0) or np.all(window == 255)):
                    noise_map[i-1, j-1] = 1
                    continue

            w_max = np.max(window)
            w_min = np.min(window)
            w_mean = np.mean(window)
            d_ij = (1.0 / 3.0) * np.sqrt(np.sum((window - w_mean) ** 2))
            
            if center == w_max or center == w_min or abs(center - w_mean) > d_ij:
                noise_map[i-1, j-1] = 1
                
    return noise_map

def adaptive_weighted_median_filter(image, noise_map):
    """
    Filtro Mediano Ponderado Adaptativo corrigido:
    Garante que a pertinência use uma referência livre de ruído 
    para não amplificar os pixels contaminados.
    """
    h, w = image.shape
    output = np.copy(image)
    pad_max = 3
    pad_img = np.pad(image, pad_max, mode='reflect').astype(np.float32)
    pad_noise = np.pad(noise_map, pad_max, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            if noise_map[i, j] == 0:
                continue

            pi, pj = i + pad_max, j + pad_max
            local_noise = np.sum(pad_noise[pi-1:pi+2, pj-1:pj+2])
            
            # Equação (9): Seleção de tamanho de janela
            if local_noise <= 3:
                fw_size = 3
            elif local_noise <= 6:
                fw_size = 5
            else:
                fw_size = 7
                
            r = fw_size // 2
            window = pad_img[pi-r:pi+r+1, pj-r:pj+r+1]
            win_noise = pad_noise[pi-r:pi+r+1, pj-r:pj+r+1]

            # Coleta apenas os pixels que NÃO são ruído na janela para servir de referência
            clean_pixels = window[win_noise == 0]
            if len(clean_pixels) > 0:
                ref_val = np.median(clean_pixels)
            else:
                ref_val = np.median(window)

            # Equações (10) e (11): Pertinência calculada em relação ao valor não ruidoso
            diff = np.abs(window - ref_val)
            membership = 1.0 / (1.0 + (diff / 255.0 * 10.0) ** 2)
            
            flat_vals = window.flatten()
            flat_memb = membership.flatten()
            sorted_idx = np.argsort(flat_memb)
            
            weighted_elements = []
            group_size = max(1, len(flat_vals) // fw_size)
            
            for idx_rank, original_idx in enumerate(sorted_idx):
                weight = min(fw_size, (idx_rank // group_size) + 1)
                weighted_elements.extend([flat_vals[original_idx]] * weight)
                
            output[i, j] = np.median(weighted_elements)
            
    return output.astype(np.uint8)

def compute_gradients_and_nms(image):
    img_float = image.astype(np.float64)
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx) * 180 / np.pi
    direction[direction < 0] += 180
    
    h, w = image.shape
    nms = np.zeros((h, w), dtype=np.float32)
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q, r = 255.0, 255.0
            angle = direction[i, j]
            
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = magnitude[i, j+1], magnitude[i, j-1]
            elif 22.5 <= angle < 67.5:
                q, r = magnitude[i+1, j-1], magnitude[i-1, j+1]
            elif 67.5 <= angle < 112.5:
                q, r = magnitude[i+1, j], magnitude[i-1, j]
            elif 112.5 <= angle < 157.5:
                q, r = magnitude[i-1, j-1], magnitude[i+1, j+1]
                
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms[i, j] = magnitude[i, j]
                
    return magnitude, nms

def iterative_global_threshold(values, tolerance=0.5):
    if len(values) == 0:
        return 0.0
    t_current = (np.max(values) + np.min(values)) / 2.0
    for _ in range(100):
        g1 = values[values > t_current]
        g2 = values[values <= t_current]
        m1 = np.mean(g1) if len(g1) > 0 else t_current
        m2 = np.mean(g2) if len(g2) > 0 else t_current
        t_next = (m1 + m2) / 2.0
        if abs(t_next - t_current) <= tolerance:
            return t_next
        t_current = t_next
    return t_current

def dual_threshold_hysteresis(nms_img):
    valid_pixels = nms_img[nms_img > 0]
    if len(valid_pixels) == 0:
        return np.zeros_like(nms_img, dtype=np.uint8)
        
    th = iterative_global_threshold(valid_pixels)
    sub_pixels = valid_pixels[valid_pixels <= th]
    tl = iterative_global_threshold(sub_pixels)
    
    h, w = nms_img.shape
    edges = np.zeros((h, w), dtype=np.uint8)
    
    edges[nms_img >= th] = 255
    edges[(nms_img >= tl) & (nms_img < th)] = 50
    
    # Histerese
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if edges[i, j] == 50:
                if np.any(edges[i-1:i+2, j-1:j+2] == 255):
                    edges[i, j] = 255
                else:
                    edges[i, j] = 0
                    
    return edges

def add_salt_and_pepper(image, ratio):
    noisy = image.copy()
    num_noise = int(ratio * image.size)
    coords_salt = [np.random.randint(0, i, num_noise // 2) for i in image.shape]
    coords_pepper = [np.random.randint(0, i, num_noise // 2) for i in image.shape]
    noisy[tuple(coords_salt)] = 255
    noisy[tuple(coords_pepper)] = 0
    return noisy

if __name__ == "__main__":
    base_dir = r"C:\opencv\sources\images"
    image_filename = "lena.png"
    image_path = os.path.join(base_dir, image_filename)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir o arquivo em: {image_path}")

    noise_ratio = 0.5
    noisy_img = add_salt_and_pepper(img, noise_ratio)
    
    # Pipeline do Artigo
    noise_mask = detect_noise(noisy_img)
    wmf_img = adaptive_weighted_median_filter(noisy_img, noise_mask)
    mag, nms_img = compute_gradients_and_nms(wmf_img)
    proposed_edges = dual_threshold_hysteresis(nms_img)
    
    # Canny Clássico para comparação
    classic_blurred = cv2.GaussianBlur(noisy_img, (5, 5), 1.4)
    classic_canny = cv2.Canny(classic_blurred, 50, 150)
    
    # Exibição dos resultados
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("1. Original")
    
    axes[0, 1].imshow(noisy_img, cmap='gray')
    axes[0, 1].set_title(f"2. Ruído Sal e Pimenta ({int(noise_ratio*100)}%)")
    
    axes[0, 2].imshow(wmf_img, cmap='gray')
    axes[0, 2].set_title("3. Filtragem WMF Adaptativa")
    
    axes[1, 0].imshow(nms_img, cmap='gray')
    axes[1, 0].set_title("4. Gradiente com NMS")
    
    axes[1, 1].imshow(classic_canny, cmap='gray')
    axes[1, 1].set_title("5. Canny Clássico")
    
    axes[1, 2].imshow(proposed_edges, cmap='gray')
    axes[1, 2].set_title("6. Canny Melhorado (Artigo)")
    
    for row in axes:
        for ax in row:
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()