"""Tile sanity PNGs into one image for quick review: python contact_sheet.py out.png uid1 uid2 ..."""
import sys
from pathlib import Path
from PIL import Image
out, uids = sys.argv[1], sys.argv[2:]
d = Path(__file__).parent / "output" / "sanity"
ims = [Image.open(d / f"{u}.png") for u in uids]
w, h = ims[0].size
s = 0.5
cols = 2
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (int(w * s) * cols, int(h * s) * rows), "white")
for i, im in enumerate(ims):
    sheet.paste(im.resize((int(w * s), int(h * s))), ((i % cols) * int(w * s), (i // cols) * int(h * s)))
sheet.save(out)
