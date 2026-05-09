import struct
from pathlib import Path
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Decode:

    def read_chunk(f):
        """
        odczytuje pojedynczy chunk z pliku PNG
    
        zwraca
            (ctype, data, length, crc):
            - ctype: 4-bajtowy typ chunku (np. b'IHDR', b'IDAT')
            - data: dane chunku o długości length
            - length: długość danych chunku
            - crc: suma kontrolna CRC32 chunku
        """
        length = struct.unpack('>I', f.read(4))[0]
        ctype = f.read(4)
        data = f.read(length)
        crc = struct.unpack('>I', f.read(4))[0]
        return ctype, data, length, crc

    def parse_ihdr(data):
        """
        parsuje dane chunku IHDR (Image Header)
        
        zwraca:
            - width: szerokość obrazu w pikselach
            - height: wysokość obrazu w pikselach
            - bit_depth: głębia bitowa (1, 2, 4, 8, 16)
            - color_type: typ koloru (0=grayscale, 2=RGB, 3=indeksowany, 4=grayscale+alpha, 6=RGBA)
            - compression_method: metoda kompresji (0 = deflate)
            - filter_method: metoda filtrowania (0 = adaptive)
            - interlace_method: metoda przeplotu (0=none, 1=Adam7)
        """
        w, h, depth, color_type, comp, filt, inter = struct.unpack('>IIBBBBB', data)
        return {
            'width': w,
            'height': h,
            'bit_depth': depth,
            'color_type': color_type,  # 0=gray, 2=RGB, 3=indeksowany, 4=gray+alpha, 6=RGBA
            'compression_method': comp,
            'filter_method': filt,
            'interlace_method': inter,
        }
    
    def fourier(photo):
        transform = fft2(photo)
        transform_shifted = fftshift(transform)
        magnitude = np.abs(transform_shifted)
        phase = np.angle(transform_shifted)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Wykres modułu (z logarytmiczną skalą dla lepszej widoczności)
        ax1.imshow(np.log(1 + magnitude), cmap='gray')
        ax1.set_title('Moduł transformacji Fouriera')
        ax1.axis('off')
        
        # Wykres fazy
        ax2.imshow(phase, cmap='gray')
        ax2.set_title('Faza transformacji Fouriera')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def test_fourier(photo):
        photo_float = photo.astype(float)
        
        transform = fft2(photo_float)
        transform_shifted = fftshift(transform)
        
        transform_unshifted = ifftshift(transform_shifted)
        reconstructed = ifft2(transform_unshifted).real
        
        # obliczenie bledu
        error = np.mean(photo_float - reconstructed)
        max_error = np.max(np.abs(photo_float - reconstructed))
        
        print("\nTest poprawnosci transformacji Fouriera:")
        print(f"  Blad: {error}")
        print(f"  Maksymalny blad bezwzgledny: {max_error}")
        
   
        if error < 1e-9 and max_error < 1e-9:
            print("  ✓ Transformacja jest poprawna (błędy w granicach tolerancji numerycznej).")
        else:
            print("  ✗ Wykryto błąd w transformacji (błędy przekraczają tolerancję).")
        
        # wyświetlenie różnicy jeśli błąd jest duży
        if error > 1e-9:
            diff = np.abs(photo_float - reconstructed)
            plt.figure(figsize=(8, 6))
            plt.imshow(diff, cmap='hot')
            plt.title('Różnica między oryginałem a odtworzonym obrazem')
            plt.colorbar()
            plt.axis('off')
            plt.show()

def import_photo(path):
    f = path.open('rb')
    signature = f.read(8)
    if signature != b'\x89PNG\r\n\x1a\n':
        print('niepoprawny format PNG')
        f.close()
        return None
    return f

def main():
    png_path = Path('shark.png')
    
    try:
        img = Image.open(png_path).convert('L') # skala szarosci
        photo = np.array(img)
    except Exception as e:
        print(f'blad ladowania obrazu: {e}')
        return
    
    file = import_photo(png_path)

    ctype, data, length, crc = Decode.read_chunk(file)
    print('read_chunk result:')
    print(f'  type: {ctype}')
    print(f'  length: {length}')
    print(f'  crc: {crc}')

    if ctype == b'IHDR':
        ihdr_info = Decode.parse_ihdr(data)
        print('\nparse_ihdr result:')
        for key, value in ihdr_info.items():
            print(f'  {key}: {value}')
    else:
        print('pierwszy chunk nie jest IHDR, a:', ctype)

    Decode.fourier(photo)
    Decode.test_fourier(photo)

if __name__ == '__main__':
    main()