import struct
from pathlib import Path
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import binascii

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
    
    def parse_text(data):
        """
        parsuje dane chunku tEXt (tekstowy)
        """
        null_pos = data.find(b'\x00')
        if null_pos == -1:
            return "Invalid tEXt chunk"
        key = data[:null_pos].decode('latin-1')
        value = data[null_pos+1:].decode('latin-1')
        return f"Klucz: {key}, Wartość: {value}"
    
    def parse_time(data):
        """
        parsuje dane chunku tIME (czas modyfikacji)
        """
        year, month, day, hour, minute, second = struct.unpack('>HBBBBB', data)
        return f"Czas modyfikacji: {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
    
    def parse_exif(data):
        """
        parsuje dane chunku eXIf (EXIF)
        """
        return f"EXIF dane (hex): {binascii.hexlify(data[:100]).decode('ascii')}..." if len(data) > 100 else f"EXIF dane (hex): {binascii.hexlify(data).decode('ascii')}"
    
    def fourier(photo):
        transform = fft2(photo)
        transform_shifted = fftshift(transform)
        magnitude = np.abs(transform_shifted)
        phase = np.angle(transform_shifted)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # wykres modulu
        ax1.imshow(np.log(1 + magnitude), cmap='gray')
        ax1.set_title('Moduł transformacji Fouriera')
        ax1.axis('off')
        
        # wykres fazy
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
        error = np.abs(photo_float - reconstructed)
        max_error = np.max(np.abs(photo_float - reconstructed))
        
        print("\ntest poprawnosci transformacji Fouriera:")
        print(f"  blad: {error}")
        print(f"  maksymalny blad bezwzgledny: {max_error}")
        
   
        if np.all(error < 1e-9) and max_error < 1e-9:
            print("transformacja jest poprawna")
        else:
            print("wykryto błąd w transformacji")
        
        # wyświetlenie różnicy jeśli błąd jest duży
        if np.any(error > 1e-9):
            diff = np.abs(photo_float - reconstructed)
            plt.figure(figsize=(8, 6))
            plt.imshow(diff, cmap='hot')
            plt.title('rożnica między oryginałem a odtworzonym obrazem')
            plt.colorbar()
            plt.axis('off')
            plt.show()

def write_png_chunk(f, ctype, data):
    """
    zapisuje chunk PNG i wylicza dla niego poprawny CRC
    """
    f.write(struct.pack('>I', len(data)))
    f.write(ctype)
    f.write(data)
    crc = binascii.crc32(ctype + data) & 0xffffffff
    f.write(struct.pack('>I', crc))

def anonymize_png(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    critical_chunks = {b'IHDR', b'PLTE', b'IDAT', b'IEND'}
    removed = []

    with input_path.open('rb') as fin, output_path.open('wb') as fout:
        signature = fin.read(8)
        if signature != b'\x89PNG\r\n\x1a\n':
            print('niepoprawny format PNG')
            return None

        fout.write(signature)

        while True:
            ctype, data, length, crc = Decode.read_chunk(fin)

            if ctype not in critical_chunks:
                removed.append(ctype.decode('ascii'))
            else:
                write_png_chunk(fout, ctype, data)

            if ctype == b'IEND':
                break

    print(f'usuniete chunki: {removed if removed else "brak"}')
    return output_path

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
    
    # wyswietlanie obraz
    plt.figure(figsize=(8, 6))
    plt.imshow(photo, cmap='gray')
    plt.title('Obraz PNG')
    plt.axis('off')
    plt.show()
    
    file = import_photo(png_path)
    if not file:
        return

    chunks = []
    while True:
        ctype, data, length, crc = Decode.read_chunk(file)
        chunks.append((ctype, data, length, crc))
        if ctype == b'IEND':
            break
    
    file.close()
    
    # wyswietlanie informacji o chunkach
    print("informacje o chunkach:")
    ancillary_displayed = 0
    for ctype, data, length, crc in chunks:
        ctype_str = ctype.decode('ascii')
        print(f"\nChunk: {ctype_str}, Długość: {length}")
        
        if ctype == b'IHDR':
            ihdr_info = Decode.parse_ihdr(data)
            print("  Atrybuty IHDR:")
            for key, value in ihdr_info.items():
                print(f"    {key}: {value}")
        elif ctype == b'PLTE':
            print(f"  Critical chunk PLTE: surowe bajty (hex): {binascii.hexlify(data[:100]).decode('ascii')}..." if len(data) > 100 else f"  Critical chunk PLTE: surowe bajty (hex): {binascii.hexlify(data).decode('ascii')}")
        elif ctype in {b'IDAT', b'IEND'}:
            print(f"  Critical chunk {ctype_str}: dane binarne, długość {length}")
        elif ctype == b'tEXt' and ancillary_displayed < 3:
            print(f"  Ancillary chunk tEXt: {Decode.parse_text(data)}")
            ancillary_displayed += 1
        elif ctype == b'tIME' and ancillary_displayed < 3:
            print(f"  Ancillary chunk tIME: {Decode.parse_time(data)}")
            ancillary_displayed += 1
        elif ctype == b'eXIf' and ancillary_displayed < 3:
            print(f"  Ancillary chunk eXIf: {Decode.parse_exif(data)}")
            ancillary_displayed += 1
        else:
            if ctype_str[0].islower():  # ancillary
                print(f"  Ancillary chunk {ctype_str}: dane binarne, długość {length}")
    
    # transformacja Fouriera
    Decode.fourier(photo)
    Decode.test_fourier(photo)

    # anonimizacja
    anonymize_png("shark.png", "anonimized.png")

if __name__ == '__main__':
    main()