"""Vectorize the pop-art image into flat-color layers (potrace per color)."""
import subprocess, sys, re, os
import numpy as np
from PIL import Image, ImageFilter

def rgb2lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)
from scipy import ndimage

SRC = sys.argv[1]
OUT = os.path.dirname(os.path.abspath(__file__))
SCALE = 3

# bottom -> top. yellow is the background rect.
PALETTE = [
    ("yellow", (252, 196, 10)),
    ("cream",  (246, 214, 178)),
    ("blue",   (31, 74, 112)),
    ("red",    (228, 46, 20)),
    ("black",  (30, 26, 24)),
]
# extra sample colors that should map onto a palette ink
ALIASES = [("red", (238, 90, 40)), ("black", (60, 46, 38)), ("cream", (242, 185, 145)),
           ("blue", (60, 100, 140))]

img = Image.open(SRC).convert("RGB")
W, H = img.size
big = img.resize((W*SCALE, H*SCALE), Image.LANCZOS).filter(ImageFilter.MedianFilter(3))
lab = rgb2lab(np.asarray(big) / 255.0)

names = [n for n, _ in PALETTE]
refs = PALETTE + ALIASES
ref_lab = rgb2lab(np.array([[c for _, c in refs]]) / 255.0)[0]
d = ((lab[..., None, :] - ref_lab[None, None]) ** 2).sum(-1)
idx = np.array([names.index(n) for n, _ in refs])[d.argmin(-1)]

# remove speckle: tiny islands get absorbed by majority filter
for _ in range(1):
    counts = np.stack([ndimage.uniform_filter((idx == i).astype(np.float32), 3) for i in range(len(names))])
    idx = counts.argmax(0)

def trace(mask, name, hexcol):
    pbm = os.path.join(OUT, f"_tmp_{name}.pbm")
    Image.fromarray(((~mask) * 255).astype(np.uint8)).convert("1").save(pbm)
    svg = subprocess.run(["potrace", pbm, "-s", "-o", "-", "-t", "4", "-a", "1.0", "-O", "0.4",
                          "--flat", "-C", hexcol], capture_output=True, text=True, check=True).stdout
    os.remove(pbm)
    g = re.search(r"(<g transform=.*?</g>)", svg, re.S).group(1)
    return g

CREAM, RED, BLACK = names.index("cream"), names.index("red"), names.index("black")

def disk(r):
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return x*x + y*y <= r*r

# --- split red into solid areas (dress, lips) and halftone dots (skin) ---
red = idx == RED
solid = ndimage.binary_propagation(ndimage.binary_opening(red, disk(9)), mask=red)
dots = red & ~solid
# skin = areas dense with dots; fill to a solid region, never over black/solid red
skin = ndimage.uniform_filter(dots.astype(np.float32), 45) > 0.06
skin &= (idx == CREAM) | dots
skin = ndimage.binary_closing(skin, disk(6))
skin = ndimage.binary_opening(skin, disk(4)) & ~solid
lbl, n = ndimage.label(skin)
sizes = ndimage.sum(skin, lbl, range(1, n+1))
skin = np.isin(lbl, 1 + np.flatnonzero(sizes > 3000))

# dot coverage measured from the source, inside the skin core
core = ndimage.binary_erosion(skin, disk(12))
coverage = dots[core].mean()
SPACING = 6.7 * SCALE                  # lattice pitch along the 45-degree axes
a = SPACING / np.sqrt(2)               # horizontal/vertical step of the staggered grid
R = np.sqrt(coverage * SPACING**2 / np.pi)
print(f"dot coverage {coverage:.2f}, pitch {SPACING:.1f}, radius {R:.2f}")

reach = ndimage.binary_dilation(skin, disk(int(R) + 2))
circles = []
for n_ in range(int(H*SCALE / a) + 2):
    for m in range(int(W*SCALE / a) + 2):
        if (m + n_) % 2: continue
        x, y = m * a, n_ * a
        xi, yi = int(x), int(y)
        if yi < H*SCALE and xi < W*SCALE and reach[yi, xi]:
            circles.append(f"M{x-R:.2f} {y:.2f}a{R:.2f} {R:.2f} 0 1 0 {2*R:.2f} 0a{R:.2f} {R:.2f} 0 1 0 {-2*R:.2f} 0z")
print("dots:", len(circles))

def trace_path(mask, name):
    g = trace(mask, name, "#000000")
    tf = re.search(r'transform="([^"]+)"', g).group(1)
    d = " ".join(re.findall(r'd="([^"]+)"', g))
    return tf, d

layers = []
for i, (name, rgb) in enumerate(PALETTE):
    hexcol = "#%02x%02x%02x" % rgb
    if name == "yellow":
        body = f'<rect width="{W*SCALE}" height="{H*SCALE}" fill="{hexcol}"/>'
    else:
        if name == "red":
            mask = solid
        elif name == "cream":
            mask = (idx == CREAM) | dots | skin
        else:
            mask = idx == i
        if name != "black":  # grow fills under the black linework to avoid seams
            mask = mask | (ndimage.binary_dilation(mask, iterations=2) & (idx > i))
        body = trace(mask, name, hexcol)
        if name == "red":
            tf, d = trace_path(skin | (ndimage.binary_dilation(skin, iterations=3) & (idx == BLACK)), "skin")
            body += (f'<defs><clipPath id="skin-clip"><path transform="{tf}" d="{d}"/></clipPath></defs>'
                     f'<g id="halftone" clip-path="url(#skin-clip)"><path fill="{hexcol}" d="{"".join(circles)}"/></g>')
    layers.append((name, hexcol, body))
    single = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W*SCALE} {H*SCALE}" '
              f'width="{W}" height="{H}">{body}</svg>')
    open(os.path.join(OUT, f"claude-layer-{i+1}-{name}.svg"), "w").write(single)
    print("traced", name)

groups = "\n".join(
    f'<g id="{n}" inkscape:groupmode="layer" inkscape:label="{n} {c}">{b}</g>' for n, c, b in layers)
full = (f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'viewBox="0 0 {W*SCALE} {H*SCALE}" width="{W*2}" height="{H*2}">\n{groups}\n</svg>')
open(os.path.join(OUT, "claude-booty-move-layered.svg"), "w").write(full)
print("done")
