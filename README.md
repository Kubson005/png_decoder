## instalacja

```bash
pip install -r requirements.txt
```

## funkcje programu

# odczyt chunków PNG
- Typ chunku (np. IHDR, IDAT)
- Dane chunku
- Długość danych
- Sumę kontrolną CRC32

# parsowanie naglowka (IHDR)
- Szerokość i wysokość w pikselach
- Głębia bitowa (1, 2, 4, 8, 16)
- Typ koloru (grayscale, RGB, indeksowany, RGBA)
- Metody: kompresji, filtrowania, przeplotu

# analiza Fouriera
- Moduł transformacji Fouriera (widmo amplitudowe)
- Fazę transformacji Fouriera (widmo fazowe)