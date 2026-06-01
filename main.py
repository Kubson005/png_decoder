import binascii
import struct
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift, ifft2, ifftshift

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CRITICAL_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
CHRM_LABELS = (
    "white_x",
    "white_y",
    "red_x",
    "red_y",
    "green_x",
    "green_y",
    "blue_x",
    "blue_y",
)

class Decode:
    @staticmethod
    def read_chunk(file_obj):
        length_bytes = file_obj.read(4)
        if len(length_bytes) == 0:
            return None
        if len(length_bytes) != 4:
            raise ValueError("niepelna dlugosc chunku")

        length = struct.unpack(">I", length_bytes)[0]
        ctype = file_obj.read(4)
        data = file_obj.read(length)
        crc_bytes = file_obj.read(4)

        if len(ctype) != 4 or len(data) != length or len(crc_bytes) != 4:
            raise ValueError("niepelny chunk PNG")

        crc = struct.unpack(">I", crc_bytes)[0]
        return ctype, data, length, crc

    @staticmethod
    def parse_ihdr(data):
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
    def parse_text(data):
        null_pos = data.find(b"\x00")
        if null_pos == -1:
            return "niepoprawny tEXt"
        key = data[:null_pos].decode("latin-1", errors="replace")
        value = data[null_pos + 1 :].decode("latin-1", errors="replace")
        return f"klucz={key}, wartosc={value}"

    @staticmethod
    def parse_time(data):
        year, month, day, hour, minute, second = struct.unpack(">HBBBBB", data)
        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"

    @staticmethod
    def parse_phys(data):
        px_x, px_y, unit = struct.unpack(">IIB", data)
        if unit == 1:
            return f"X={px_x} px/m, Y={px_y} px/m (jednostka: metr)"
        return f"X={px_x}, Y={px_y} (jednostka: nieokreslona)"

    @staticmethod
    def parse_gama(data):
        gamma_raw = struct.unpack(">I", data)[0]
        return f"gamma={gamma_raw / 100000:.5f}"

    @staticmethod
    def parse_chrm(data):
        vals = struct.unpack(">IIIIIIII", data)
        parsed = {label: value / 100000 for label, value in zip(CHRM_LABELS, vals)}
        return ", ".join(f"{k}={v:.5f}" for k, v in parsed.items())

    @staticmethod
    def parse_exif(data):
        hex_data = chunk_to_hex(data)
        if len(data) < 8:
            return f"eXIf (hex): {hex_data}"

        byte_order = data[:2]
        if byte_order not in (b"II", b"MM"):
            return f"eXIf (hex): {hex_data}"

        endian = "<" if byte_order == b"II" else ">"
        marker = struct.unpack(endian + "H", data[2:4])[0]
        ifd_offset = struct.unpack(endian + "I", data[4:8])[0]
        return (
            f"TIFF byte_order={byte_order.decode('ascii')}, "
            f"marker=0x{marker:04x}, ifd_offset={ifd_offset}"
        )

    @staticmethod
    def fourier(photo):
        transform = fftshift(fft2(photo.astype(float)))
        magnitude = np.log1p(np.abs(transform))
        phase = np.angle(transform)

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(magnitude, cmap="gray")
        ax1.set_title("Modul transformacji Fouriera")
        ax1.axis("off")

        ax2.imshow(phase, cmap="gray")
        ax2.set_title("Faza transformacji Fouriera")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def test_fourier(photo):
        photo_float = photo.astype(float)
        shifted = fftshift(fft2(photo_float))
        reconstructed = ifft2(ifftshift(shifted)).real

        error = np.abs(photo_float - reconstructed)
        max_error = float(np.max(error))
        mean_error = float(np.mean(error))

        print("\ntest poprawnosci transformacji Fouriera:")
        print(f"  maksymalny blad bezwzgledny: {max_error}")
        print(f"  sredni blad bezwzgledny: {mean_error}")
        print("  wynik:", "transformacja poprawna" if max_error < 1e-9 else "blad transformacji")


def chunk_to_hex(data):
    return binascii.hexlify(data).decode("ascii")

def shorten_text(text, max_len=160):
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return f"{text[:half]} ... {text[-half:]}"

def is_ancillary(ctype):
    return bool(ctype[0] & 0x20)

SELECTED_ANCILLARY = {
    b"tEXt": Decode.parse_text,
    b"tIME": Decode.parse_time,
    b"cHRM": Decode.parse_chrm,
    b"eXIf": Decode.parse_exif,
    b"pHYs": Decode.parse_phys,
    b"gAMA": Decode.parse_gama,
}


def chunk_data(chunks, chunk_type):
    return next((data for ctype, data, _, _ in chunks if ctype == chunk_type), None)

def print_step(number, title):
    print(f"\n[{number}] {title}")

def read_png(path):
    chunks = []
    with path.open("rb") as file_obj:
        signature = file_obj.read(8)
        if signature != PNG_SIGNATURE:
            raise ValueError("niepoprawny format PNG")

        while True:
            chunk = Decode.read_chunk(file_obj)
            if chunk is None:
                raise ValueError("brak chunku IEND")
            chunks.append(chunk)
            if chunk[0] == b"IEND":
                break

        trailing_data = file_obj.read()
    return chunks, trailing_data


def print_file_attributes(path, chunks):
    print("atrybuty pliku:")
    print(f"  nazwa: {path.name}")
    print(f"  rozmiar pliku: {path.stat().st_size} bajtow")

    ihdr = chunk_data(chunks, b"IHDR")
    if ihdr is not None:
        parsed = Decode.parse_ihdr(ihdr)
        print("  naglowek IHDR:")
        for key, value in parsed.items():
            print(f"    {key}: {value}")

    phys = chunk_data(chunks, b"pHYs")
    if phys is not None:
        print(f"  czestotliwosc probkowania (pHYs): {Decode.parse_phys(phys)}")


def print_critical_chunks(chunks):
    print("\nobowiazkowe segmenty (critical chunks):")
    for ctype, data, length, crc in chunks:
        if ctype not in CRITICAL_CHUNKS:
            continue
        name = ctype.decode("ascii")
        full_hex = chunk_to_hex(data)
        shown_hex = shorten_text(full_hex, max_len=200)
        print(f"\nchunk: {name}")
        print(f"  dlugosc: {length}")
        print(f"  crc: 0x{crc:08x}")
        print(f"  dane hex (podglad): {shown_hex}")
        if len(shown_hex) != len(full_hex):
            print(f"  uwaga: skrócono podglad (pelna dlugosc hex: {len(full_hex)} znakow)")
        if ctype == b"IHDR":
            parsed = Decode.parse_ihdr(data)
            print("  interpretacja IHDR:")
            for key, value in parsed.items():
                print(f"    {key}: {value}")


def print_selected_ancillary(chunks):
    print("\nwybrane dodatkowe segmenty (ancillary chunks):")

    shown_types = set()
    for ctype, data, _, _ in chunks:
        parser = SELECTED_ANCILLARY.get(ctype)
        if parser is None:
            continue
        shown_types.add(ctype)
        print(f"  {ctype.decode('ascii')}: {parser(data)}")

    print(f"  liczba roznych typow ancillary pokazanych: {len(shown_types)}")
    if len(shown_types) < 3:
        print("  uwaga: ten plik zawiera mniej niz 3 wybrane typy ancillary")


def anonymize_png(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    removed_chunks = []
    with input_path.open("rb") as fin, output_path.open("wb") as fout:
        signature = fin.read(8)
        if signature != PNG_SIGNATURE:
            raise ValueError("niepoprawny format PNG")
        fout.write(signature)

        while True:
            chunk = Decode.read_chunk(fin)
            if chunk is None:
                raise ValueError("brak chunku IEND")

            ctype, data, length, crc = chunk
            if is_ancillary(ctype):
                removed_chunks.append(ctype.decode("ascii"))
            else:
                fout.write(struct.pack(">I", length))
                fout.write(ctype)
                fout.write(data)
                fout.write(struct.pack(">I", crc))

            if ctype == b"IEND":
                break

        trailing_removed = len(fin.read())

    print("\nanonimizacja:")
    print(f"  usuniete ancillary: {removed_chunks if removed_chunks else 'brak'}")
    print(f"  usuniete dane po IEND (offsety/dodatki): {trailing_removed} bajtow")
    print(f"  zapisano: {output_path}")


def main():
    png_path = Path("shark.png")
    output_path = Path("anonimized.png")

    try:
        chunks, trailing = read_png(png_path)
    except Exception as error:
        print(f"blad odczytu PNG: {error}")
        return

    try:
        image = Image.open(png_path).convert("L")
        photo = np.array(image)
    except Exception as error:
        print(f"blad ladowania obrazu: {error}")
        return

    print(f"plik wejsciowy: {png_path}\n")

    print_step(1, "Ręczne dekodowanie PNG (analiza kolejnych bajtow)")
    print(f"  liczba odczytanych chunkow: {len(chunks)}")
    print("  kolejnosc chunkow:")
    for ctype, _, length, _ in chunks:
        print(f"    {ctype.decode('ascii')} (dlugosc={length})")

    print_step(2, "Atrybuty pliku (rozmiar, glebia, probkowanie, itd.)")
    print_file_attributes(png_path, chunks)

    print_step(3, "Obowiazkowe segmenty (critical) - pelna zawartosc")
    print_critical_chunks(chunks)

    print_step(4, "Wybrane segmenty dodatkowe (ancillary, min. 3 typy)")
    print_selected_ancillary(chunks)

    print_step(5, "Prezentacja obrazu")
    plt.figure(figsize=(8, 6))
    plt.imshow(photo, cmap="gray")
    plt.title("Obraz PNG")
    plt.axis("off")
    plt.show()

    print_step(6, "Widmo Fouriera (modul i faza)")
    Decode.fourier(photo)

    print_step(7, "Sposob testowania poprawnosci transformacji Fouriera")
    Decode.test_fourier(photo)

    print_step(8, "Anonimizacja bez ingerencji w obraz")
    print(f"  dane po IEND w wejsciu: {len(trailing)} bajtow")
    anonymize_png(png_path, output_path)

    try:
        out_chunks, out_trailing = read_png(output_path)
    except Exception as error:
        print(f"blad weryfikacji anonimizacji: {error}")
        return

    out_ancillary = [c.decode("ascii") for c, _, _, _ in out_chunks if is_ancillary(c)]
    out_critical = [c.decode("ascii") for c, _, _, _ in out_chunks if c in CRITICAL_CHUNKS]
    print("  weryfikacja pliku po anonimizacji:")
    print(f"    critical pozostaly: {out_critical}")
    print(f"    ancillary pozostale: {out_ancillary if out_ancillary else 'brak'}")
    print(f"    dane po IEND: {len(out_trailing)} bajtow")


if __name__ == "__main__":
    main()
