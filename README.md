# TugasMiniProjectPCV2
# MP2 - Object Counting (Parkiran Aerial)

Nama  :Ilan Hawwari Prasojo 
NRP   :5024241039

---

## Jumlah Mobil Terdeteksi: 30 Mobil

---

## Cara Menjalankan

```bash
pip install opencv-python numpy matplotlib
python counting.py
```

Pastikan gambar input ada di folder `input/parking.jpg`. Output akan otomatis tersimpan di `output/`.

---

## Penjelasan Pipeline

Pendekatan yang dipakai adalah hybrid — gabungan dual threshold, morfologi, dan watershed. Alasannya karena satu teknik saja ga cukup untuk gambar ini: mobil warnanya macem-macem (putih, hitam, silver, biru tua) dan beberapa parkir nempel satu sama lain.

### Step 1 - Eksplorasi Color Space

Sebelum mulai processing, gambarnya dilihat dulu di berbagai color space (Grayscale, LAB, HSV) buat nyari mana yang paling bisa bedain mobil sama aspal. Hasilnya grayscale cukup bagus — aspal nilainya sekitar 100-145, mobil terang di atas itu dan mobil gelap di bawahnya.

![Color Space](<img width="1800" height="1000" alt="01_color_spaces" src="https://github.com/user-attachments/assets/80a1b7b6-f4aa-4fd0-ab2b-1049217488d8" />
)

---

### Step 2 - Gaussian Blur

```python
blur = cv2.GaussianBlur(gray, (7, 7), 2)
```

Blur 7x7 dipilih buat ngurangin noise tekstur aspal. Kalau kernel terlalu kecil kurang ngaruh, terlalu besar ntar tepi mobilnya ikut blur juga.

![Blur](<img width="1000" height="600" alt="02_blur" src="https://github.com/user-attachments/assets/3ff4ee67-bd8d-4326-971e-f52ec2776b58" />
)

---

### Step 3 - Dual Thresholding

```python
_, thresh_terang = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresh_gelap  = cv2.threshold(blur, 72, 255, cv2.THRESH_BINARY_INV)
gabungan = cv2.bitwise_or(thresh_terang, thresh_gelap)
```

Pakai dua threshold karena:
- **Otsu** → nangkep mobil terang (putih/silver), threshold-nya otomatis dicari
- **Fixed inverse (72)** → nangkep mobil gelap (hitam, biru tua)
- Keduanya di-OR biar semua mobil ketangkep

![Threshold](<img width="1800" height="500" alt="03_threshold" src="https://github.com/user-attachments/assets/9ca7e5e8-5099-4a48-aa7b-b4f3760235bc" />
)

---

### Step 4 - Morfologi

```python
opened = cv2.morphologyEx(gabungan, cv2.MORPH_OPEN,  kernel_open,  iterations=2)
closed = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, kernel_close)
```

- **Opening (9x9, 2x)** → bersihin garis marka parkir yang tipis sama noise kecil
- **Closing (25x25)** → nutup lubang di dalam body mobil karena kaca/atap sering beda warna sama bodi

Kernel ellipse dipilih karena lebih natural untuk bentuk mobil dibanding kotak.

![Morfologi](<img width="1800" height="500" alt="04_morfologi" src="https://github.com/user-attachments/assets/8a8765f3-0fbf-4210-8f05-967c229d3d0f" />
)

---

### Step 5 - Watershed

Setelah closing, beberapa mobil yang parkir berdempetan masih nyambung jadi satu blob. Watershed dipake buat misahinnya.

Caranya:
1. Hitung distance transform — piksel yang jauh dari tepi blob nilainya tinggi (= pusat objek)
2. Ambil puncak-puncaknya sebagai "seed" / titik awal tiap mobil
3. Watershed "tumbuh" dari seed itu dan potong di mana dua seed ketemu

```python
dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
markers = cv2.watershed(img.copy(), markers)
```

![Watershed](<img width="1800" height="500" alt="05_watershed" src="https://github.com/user-attachments/assets/a3c27cce-a26d-4961-a1fd-42b1e5f1af6f" />
)

---

### Step 6 - Connected Components + Filter

Dari hasil watershed dibuat mask binary, terus dihitung pakai `connectedComponentsWithStats`. Tiap komponen difilter berdasarkan:

| Filter | Nilai | Alasan |
|--------|-------|--------|
| Min area | 0.08% dari total pixel | buang noise & garis sisa |
| Max area | 2% dari total pixel | buang blob yang masih nyambung |
| Max rasio sisi | 4.5 | buang objek memanjang (bukan mobil) |

Yang lolos filter dikasih bounding box hijau + nomor.

---

## Visualisasi Hasil

![Perbandingan](<img width="2000" height="900" alt="06_perbandingan" src="https://github.com/user-attachments/assets/070c9e97-e37b-4335-9ee6-a4301b2a9372" />
)

---

## Analisis

**Kendala:**

- Beberapa mobil di sisi kanan yang parkir sangat rapat masih terhitung sebagai satu meskipun sudah pakai watershed. Ini karena celah antar mobilnya hampir nol jadi distance transform-nya ga bisa bikin dua puncak yang terpisah.

- Garis parkir warna putih sempat ikut terhitung di thresh_terang, tapi berhasil dibuang sama opening karena garisnya tipis.

- Mobil yang punya kontras internal tinggi (kaca sangat gelap, bodi putih) butuh closing kernel yang besar, tapi kernel besar juga bikin blob mobil berdekatan makin mudah nyambung. Jadi agak tradeoff.

**Akurasi:**

Estimasi manual di gambar sekitar 31 mobil. Program ngedeteksi 30, jadi akurasinya sekitar 90-95%. Kemungkinan 1-2 pasang mobil yang sangat rapat terhitung sebagai satu.

## Struktur Folder

```
mp2-object-counting/
├── README.md
├── counting.py
├── input/
│   └── parking.jpg
└── output/
    ├── result.png
    └── Hasil Olah/
        ├── 01_color_spaces.png
        ├── 02_blur.png
        ├── 03_threshold.png
        ├── 04_morfologi.png
        ├── 05_watershed.png
        └── 06_perbandingan.png
```
