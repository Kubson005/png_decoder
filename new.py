import binascii
import struct
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift, ifft2, ifftshift

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# chunki krytyczne wymagane do poprawnego dekodowania PNG
CRITICAL_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}

class Decode:
    # budowa chunka: length, type, data, crc
    @staticmethod
    def read_chunk(file_obj):
        """
        odczytuje chunk PNG z pliku binarnego
        """
        # 4 bajty długości pola data w chunku
        length_bytes = file_obj.read(4)

        # pusty odczyt oznacza koniec pliku
        if len(length_bytes) == 0:
            return None
        
        # inna długość niż 4 to uszkodzona struktura
        if len(length_bytes) != 4:
            raise ValueError("niepelna dlugosc chunku")

        # big-endian (">I")
        length = struct.unpack(">I", length_bytes)[0]



        # typ chunku- 4 bajty ASCII
        ctype = file_obj.read(4)


        # odczyt dokładnie tylu bajtów, ile mówi długość chunku
        data = file_obj.read(length)



        # CRC też ma stałe 4 bajty
        crc_bytes = file_obj.read(4)

        if len(ctype) != 4 or len(data) != length or len(crc_bytes) != 4:
            raise ValueError("niepelny chunk PNG")

        # odczyt CRC jako liczby
        crc = struct.unpack(">I", crc_bytes)[0]

        return ctype, data, length, crc

    # krytyczny
    @staticmethod
    def parse_ihdr(data):
        """
        IHDR zawiera podstawowe metadane potrzebne do opisu pliku:
        rozmiar, głębię, typ koloru i parametry kodowania.
        """
        # 13 bajtów IHDR: width, height..
        w, h, depth, color_type, comp, filt, inter = struct.unpack(">IIBBBBB", data)
        return {
            "width": w,
            "height": h,
            "bit_depth": depth,
            "color_type": color_type,
            "compression_method": comp,
            "filter_method": filt,
            "interlace_method": inter,
        }

    @staticmethod
    def parse_time(data):
        """
        chunk tIME (czas modyfikacji pliku)
        """
        # tIME: rok(2B) + miesiąc/dzień/godzina/minuta/sekunda.
        year, month, day, hour, minute, second = struct.unpack(">HBBBBB", data)
        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"

    @staticmethod
    def parse_phys(data):
        """
        chunk pHYs (gęstość pikseli / proporcje pikseli)
        """
        # pHYs: 4B X, 4B Y, 1B jednostki
        px_x, px_y, unit = struct.unpack(">IIB", data)
        if unit == 1:
            return f"X={px_x} px/m, Y={px_y} px/m "
        return f"X={px_x}, Y={px_y} "

    @staticmethod
    def parse_gama(data):
        """
        chunk gAMA

        w PNG gamma jest zapisana jako liczba całkowita * 100000,
        więc dzielimy przez 100000, aby odzyskać wartość rzeczywistą
        """
        # gAMA to jedna liczba uint32
        gamma_raw = struct.unpack(">I", data)[0]
        return f"gamma={gamma_raw / 100000:.5f}"

    @staticmethod
    def parse_exif(data):
        """
        nagłówek TIFF osadzony w chunku eXIf
        """
        # HEX jako fallback, gdy danych EXIF nie da się sensownie zinterpretować
        hex_data = chunk_to_hex(data)
        if len(data) < 8:
            return f"eXIf (hex): {hex_data}"

        # nagłówek TIFF określa kolejność bajtów: "II" (little) lub "MM" (big)
        byte_order = data[:2]
        if byte_order not in (b"II", b"MM"):
            return f"eXIf (hex): {hex_data}"

        # budujemy prefiks formatu struct zgodny z kolejnością bajtów TIFF
        endian = "<" if byte_order == b"II" else ">"

        # marker TIFF (zwykle 42) i offset do pierwszego IFD.
        marker = struct.unpack(endian + "H", data[2:4])[0]
        ifd_offset = struct.unpack(endian + "I", data[4:8])[0]
        return (
            f"TIFF byte_order={byte_order.decode('ascii')}, "
            f"marker=0x{marker:04x}, ifd_offset={ifd_offset}"
        )

    @staticmethod
    def fourier(photo):
        """
        transformata Fouriera: moduł i faza

        - FFT2: przejście z dziedziny przestrzennej do częstotliwości
        - fftshift: przesunięcie do środka
        """

        # konwersja do float ogranicza błędy numeryczne przy FFT
        transform = fftshift(fft2(photo))

        # moduł liczby zespolonej opisuje energię danej częstotliwości
        magnitude = np.abs(transform)

        # skala logarytmiczna: skladowa DC w srodku jest o rzedy wielkosci
        # wieksza od reszty, wiec w skali liniowej widmo to tylko jasna kropka.
        magnitude_log = np.log1p(magnitude)

        # argument liczby zespolonej to faza widma
        phase = np.angle(transform)

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(magnitude_log, cmap="gray")
        ax1.set_title("Modul transformacji Fouriera (skala log)")
        ax1.axis("off")

        ax2.imshow(phase, cmap="gray")
        ax2.set_title("Faza transformacji Fouriera")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def test_fourier(photo):
        """
        testuje poprawność transformacji Fouriera przez rekonstrukcję obrazu
        """
        shifted = fftshift(fft2(photo))

        # odwrotna kolejność operacji: ifftshift -> ifft2
        reconstructed = ifft2(ifftshift(shifted)).real

        # błąd bezwzględny
        error = np.abs(photo - reconstructed)

        max_error = float(np.max(error))
        mean_error = float(np.mean(error))

        print("\ntest poprawnosci transformacji Fouriera:")
        print(f"  maksymalny blad bezwzgledny: {max_error}")
        print(f"  sredni blad bezwzgledny: {mean_error}")
        print("  wynik:", "transformacja poprawna" if max_error < 1e-9 else "blad transformacji")

def chunk_to_hex(data):
    """konwertuje dane binarne chunku na zapis HEX"""
    return binascii.hexlify(data).decode("ascii")

def shorten_text(text, max_len=160):
    """
    skracanie tekstu, zostawiając początek i koniec
    """
    if len(text) <= max_len:
        return text

    half = max_len // 2
    return f"{text[:half]} ... {text[-half:]}"

def is_ancillary(ctype):
    """
    rozpoznaje, czy chunk jest ancillary.
    mała litera -> ancillary, wielka litera -> critical.
    """
    return bool(ctype[0] & 0x20)

SELECTED_ANCILLARY = {
    b"tIME": Decode.parse_time,
    b"eXIf": Decode.parse_exif,
    b"pHYs": Decode.parse_phys,
    b"gAMA": Decode.parse_gama,
}

def chunk_data(chunks, chunk_type):
    """zwraca dane pierwszego chunku danego typu"""
    # next(..., None) zwraca pierwszy pasujący element
    return next((data for ctype, data, _, _ in chunks if ctype == chunk_type), None)

def read_png(path):
    """
    wczytuje PNG  i zwraca listę chunków + dane po IEND
    """
    
    chunks = []
    with path.open("rb") as file_obj:
        # każdy PNG musi zaczynać się od magic number
        signature = file_obj.read(8)
        if signature != PNG_SIGNATURE:
            raise ValueError("niepoprawny format PNG")

        while True:
            # odczyt każdego kolejnego chunku
            chunk = Decode.read_chunk(file_obj)
            if chunk is None:
                raise ValueError("brak chunku IEND")
            chunks.append(chunk)
            # IEND formalnie kończy PNG
            if chunk[0] == b"IEND":
                break

        # cokolwiek po IEND traktujemy jako dodatkowe dane
        trailing_data = file_obj.read()
    return chunks, trailing_data

def print_file_attributes(path, chunks):
    """
    wypisuje podstawowe atrybuty pliku i metadane nagłówkowe
    """
    print("atrybuty pliku:")
    print(f"  nazwa: {path.name}")
    print(f"  rozmiar pliku: {path.stat().st_size} bajtow")

    # szukamy kluczowego nagłówka IHDR
    ihdr = chunk_data(chunks, b"IHDR")
    if ihdr is not None:
        parsed = Decode.parse_ihdr(ihdr)
        print("  naglowek IHDR:")
        for key, value in parsed.items():
            print(f"    {key}: {value}")

    # pHYs jest opcjonalny, więc wyświetlamy tylko jeśli istnieje
    phys = chunk_data(chunks, b"pHYs")
    if phys is not None:
        print(f"  czestotliwosc probkowania (pHYs): {Decode.parse_phys(phys)}")

def print_critical_chunks(chunks):
    """
    wypisuje critical chunks oraz ich dane
    """
    print("\nobowiazkowe segmenty (critical chunks):")
    for ctype, data, length, crc in chunks:
        # pomijamy ancillary, bo ta sekcja dotyczy tylko chunków krytycznych
        if ctype not in CRITICAL_CHUNKS:
            continue
        name = ctype.decode("ascii")
        # konwersja na HEX, aby można było czytać dane binarne w konsoli
        full_hex = chunk_to_hex(data)
        # podgląd skrócony, aby nie zalewać terminala gigantycznym IDAT
        shown_hex = shorten_text(full_hex, max_len=200)
        print(f"\nchunk: {name}")
        print(f"  dlugosc: {length}")
        print(f"  crc: 0x{crc:08x}")
        print(f"  dane hex (podglad): {shown_hex}")
        if len(shown_hex) != len(full_hex):
            print(f" (pelna dlugosc hex: {len(full_hex)} znakow)")
        # dla IHDR dodajemy interpretację pól, nie tylko surowe bajty
        if ctype == b"IHDR":
            parsed = Decode.parse_ihdr(data)
            print("  interpretacja IHDR:")
            for key, value in parsed.items():
                print(f"    {key}: {value}")


def print_selected_ancillary(chunks):
    """
    wypisuje wybrane ancillary chunks i ich interpretację
    """
    print("\nwybrane dodatkowe segmenty (ancillary chunks):")

    # set pozwala łatwo policzyć liczbę różnych typów ancillary
    shown_types = set()
    for ctype, data, _, _ in chunks:
        # jeśli typ nie jest na liście wybranych to pomijamy
        parser = SELECTED_ANCILLARY.get(ctype)
        if parser is None:
            continue
        shown_types.add(ctype)
        print(f"  {ctype.decode('ascii')}: {parser(data)}")

    print(f"  liczba roznych typow ancillary pokazanych: {len(shown_types)}")
    if len(shown_types) < 3:
        print("  uwaga: ten plik zawiera mniej niz 3 wybrane typy ancillary")

def anonymize_png(input_path, output_path):
    """
    Tworzy zanonimizowaną wersję PNG
    - zostawiamy tylko critical chunks
    - usuwamy ancillary chunks (metadane)
    - usuwamy dane po IEND
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # lista nazw usuniętych chunków ancillary do raportu końcowego
    removed_chunks = []
    with input_path.open("rb") as fin, output_path.open("wb") as fout:
        # anonimizowany plik zachowuje poprawny podpis PNG
        signature = fin.read(8)
        if signature != PNG_SIGNATURE:
            raise ValueError("niepoprawny format PNG")
        fout.write(signature)

        while True:
            # czytamy kolejne chunki z wejścia
            chunk = Decode.read_chunk(fin)
            if chunk is None:
                raise ValueError("brak chunku IEND")

            ctype, data, length, crc = chunk
            # ancillary usuwamy (metadane), critical zostawiamy bez zmian
            if is_ancillary(ctype):
                removed_chunks.append(ctype.decode("ascii"))
            else:
                # aapis chunku 1:1 (length, type, data, CRC) zachowuje zawartość obrazu
                fout.write(struct.pack(">I", length))
                fout.write(ctype)
                fout.write(data)
                fout.write(struct.pack(">I", crc))

            # po IEND nie powinno być kolejnych chunków PNG
            if ctype == b"IEND":
                break

        # zliczamy ewentualne dodatkowe bajty po IEND
        trailing_removed = len(fin.read())

    print("\nanonimizacja:")
    print(f"  usuniete ancillary: {removed_chunks if removed_chunks else 'brak'}")
    print(f"  usuniete dane po IEND (offsety/dodatki): {trailing_removed} bajtow")
    print(f"  zapisano: {output_path}")


def main():
    png_path = Path("earthrise.png")
    output_path = Path("anonimized.png")

    try:
        # odczyt PNG i chunków
        chunks, trailing = read_png(png_path)
    except Exception as error:
        print(f"blad odczytu PNG: {error}")
        return

    try:
        # konwersja do skali szarości upraszcza analizę FFT 
        image = Image.open(png_path).convert("L")
        # zamiana do numpy do obliczeń FFT
        photo = np.array(image)
    except Exception as error:
        print(f"blad ladowania obrazu: {error}")
        return

    print(f"\nplik wejsciowy: {png_path}\n")

    print("\n[1] Ręczne dekodowanie PNG")
    print(f"  liczba odczytanych chunkow: {len(chunks)}")
    print("  kolejnosc chunkow:")
    for ctype, _, length, _ in chunks:
        #kolejność i długości chunków 
        print(f"    {ctype.decode('ascii')} (dlugosc={length})")

    print("\n[2] Atrybuty pliku (rozmiar, glebia, probkowanie, itd.)")
    print_file_attributes(png_path, chunks)

    print("\n[3] Obowiazkowe segmenty (critical) - pelna zawartosc")
    print_critical_chunks(chunks)

    print("\n[4] Wybrane segmenty dodatkowe (ancillary, min. 3 typy)")
    print_selected_ancillary(chunks)

    print("\n[5] Prezentacja obrazu")

    plt.figure(figsize=(8, 6))
    plt.imshow(photo, cmap="gray")
    plt.title("Obraz PNG")
    plt.axis("off")
    plt.show()

    print("\n[6] Widmo Fouriera (modul i faza)")
    Decode.fourier(photo)

    print("\n[7] Sposob testowania poprawnosci transformacji Fouriera")
    Decode.test_fourier(photo)

    print("\n[8] Anonimizacja bez ingerencji w obraz")
    print(f"  dane po IEND w wejsciu: {len(trailing)} bajtow")

    anonymize_png(png_path, output_path)

if __name__ == "__main__":
    main()
