import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

img = cv2.imread("input/parking.jpg")
H, W = img.shape[:2]

os.makedirs("output/Hasil Olah", exist_ok=True)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# lihat dulu gambarnya di berbagai color space buat tau mana yang paling keliatan mobil vs aspal
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0,0].imshow(img_rgb);               axes[0,0].set_title("Original");    axes[0,0].axis("off")
axes[0,1].imshow(gray, cmap="gray");     axes[0,1].set_title("Grayscale");   axes[0,1].axis("off")
axes[0,2].imshow(lab[:,:,0], cmap="gray");  axes[0,2].set_title("LAB - L"); axes[0,2].axis("off")
axes[1,0].imshow(hsv[:,:,2], cmap="gray");  axes[1,0].set_title("HSV - V"); axes[1,0].axis("off")
axes[1,1].imshow(lab[:,:,1], cmap="RdYlGn"); axes[1,1].set_title("LAB - A"); axes[1,1].axis("off")
axes[1,2].imshow(lab[:,:,2], cmap="coolwarm"); axes[1,2].set_title("LAB - B"); axes[1,2].axis("off")
plt.suptitle("Eksplorasi Color Space", fontsize=14)
plt.tight_layout()
plt.savefig("output/Hasil Olah/01_color_spaces.png", dpi=100)
plt.close()

# blur dulu biar noise aspal ilang, pake 7x7 karena kalo kecil kurang ngaruh
blur = cv2.GaussianBlur(gray, (7, 7), 2)

plt.figure(figsize=(10, 6))
plt.imshow(blur, cmap="gray")
plt.title("Setelah Gaussian Blur 7x7")
plt.axis("off")
plt.savefig("output/Hasil Olah/02_blur.png", dpi=100)
plt.close()

# masalahnya mobil ada yang putih/silver dan ada yang item/gelap
# satu threshold ga cukup, jadi pisah dua
otsu_val, thresh_terang = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresh_gelap = cv2.threshold(blur, 72, 255, cv2.THRESH_BINARY_INV)

gabungan = cv2.bitwise_or(thresh_terang, thresh_gelap)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(thresh_terang, cmap="gray"); axes[0].set_title(f"Otsu (thr={otsu_val:.0f}) - mobil terang"); axes[0].axis("off")
axes[1].imshow(thresh_gelap,  cmap="gray"); axes[1].set_title("Inv thr=72 - mobil gelap");                  axes[1].axis("off")
axes[2].imshow(gabungan,      cmap="gray"); axes[2].set_title("Gabungan");                                   axes[2].axis("off")
plt.suptitle("Dual Thresholding", fontsize=13)
plt.tight_layout()
plt.savefig("output/Hasil Olah/03_threshold.png", dpi=100)
plt.close()

# opening buat buang garis parkir & noise kecil
# closing buat nutup lubang di dalam body mobil (kaca, atap)
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

opened = cv2.morphologyEx(gabungan, cv2.MORPH_OPEN,  kernel_open,  iterations=2)
closed = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, kernel_close)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(gabungan, cmap="gray"); axes[0].set_title("Sebelum Morfologi"); axes[0].axis("off")
axes[1].imshow(opened,   cmap="gray"); axes[1].set_title("Setelah Opening");   axes[1].axis("off")
axes[2].imshow(closed,   cmap="gray"); axes[2].set_title("Setelah Closing");   axes[2].axis("off")
plt.suptitle("Morfologi", fontsize=13)
plt.tight_layout()
plt.savefig("output/Hasil Olah/04_morfologi.png", dpi=100)
plt.close()

# watershed buat misahin mobil yang nempel satu sama lain setelah closing
dist = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

_, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

sure_bg = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=4)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(img.copy(), markers)

ws_vis = img_rgb.copy()
ws_vis[markers == -1] = [255, 40, 40]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].imshow(dist_vis, cmap="plasma"); axes[0].set_title("Distance Transform");    axes[0].axis("off")
axes[1].imshow(sure_fg,  cmap="gray");   axes[1].set_title("Seed (Sure Foreground)"); axes[1].axis("off")
axes[2].imshow(ws_vis);                  axes[2].set_title("Hasil Watershed");        axes[2].axis("off")
plt.suptitle("Watershed Segmentation", fontsize=13)
plt.tight_layout()
plt.savefig("output/Hasil Olah/05_watershed.png", dpi=100)
plt.close()

# bikin mask dari hasil watershed trus connected components buat hitung
mask_akhir = np.zeros((H, W), dtype=np.uint8)
mask_akhir[markers > 1] = 255

jumlah_label, label_map, stats, centroids = cv2.connectedComponentsWithStats(mask_akhir, connectivity=8)

total_px  = H * W
min_area  = int(total_px * 0.0008)   # terlalu kecil = noise / garis
max_area  = int(total_px * 0.020)    # terlalu besar = beberapa mobil masih nyambung
max_rasio = 4.5                       # bentuk terlalu panjang = bukan mobil

print(f"filter area: {min_area} - {max_area} px")

hasil = img_rgb.copy()
n_mobil = 0

for i in range(1, jumlah_label):
    area = stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    if w < 1 or h < 1:
        continue

    rasio = max(w, h) / max(min(w, h), 1)

    if min_area < area < max_area and rasio < max_rasio:
        n_mobil += 1
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])

        cv2.rectangle(hasil, (x, y), (x + w, y + h), (30, 220, 60), 4)
        cv2.circle(hasil, (cx, cy), 8, (220, 40, 40), -1)
        cv2.putText(hasil, str(n_mobil), (x + 5, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 50), 2)

teks = f"Mobil terdeteksi: {n_mobil}"
cv2.putText(hasil, teks, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 0),    6)
cv2.putText(hasil, teks, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 2.0, (50, 255, 80), 3)

cv2.imwrite("output/result.png", cv2.cvtColor(hasil, cv2.COLOR_RGB2BGR))

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
axes[0].imshow(img_rgb); axes[0].set_title("Original");                  axes[0].axis("off")
axes[1].imshow(hasil);   axes[1].set_title(f"Hasil - {n_mobil} Mobil"); axes[1].axis("off")
plt.suptitle("MP2 - Car Counting", fontsize=15)
plt.tight_layout()
plt.savefig("output/Hasil Olah/06_perbandingan.png", dpi=100)
plt.close()

print(f"\nTotal mobil: {n_mobil}")
