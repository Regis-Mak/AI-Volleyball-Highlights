"""A small blocky pixel font for the nametags, in the spirit of Minecraft's
(drawn here from scratch, so no game files are needed).

Each glyph is up to 5 pixels wide and 8 rows tall: 7 rows above the baseline
and 1 below it for the tails of g, j, p, q and y. A tag is the text on a
see-through dark box, scaled up by a whole number of screen pixels per font
pixel so the edges stay crisp.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np

ROWS = 8          # 7 above the baseline + 1 below
PAD = 1           # box padding around the text, in font pixels

# Rows top to bottom, "#" = on. Missing rows at the bottom are blank.
GLYPHS: dict[str, str] = {
    "A": ".###.|#...#|#...#|#####|#...#|#...#|#...#",
    "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".###.|#...#|#....|#....|#....|#...#|.###.",
    "D": "####.|#...#|#...#|#...#|#...#|#...#|####.",
    "E": "#####|#....|#....|####.|#....|#....|#####",
    "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".####|#....|#....|#..##|#...#|#...#|.###.",
    "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": "###|.#.|.#.|.#.|.#.|.#.|###",
    "J": "....#|....#|....#|....#|#...#|#...#|.###.",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#",
    "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#...#|#...#|#...#|#...#",
    "N": "#...#|##..#|#.#.#|#..##|#...#|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.",
    "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "Q": ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#",
    "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".####|#....|#....|.###.|....#|....#|####.",
    "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.",
    "V": "#...#|#...#|#...#|#...#|.#.#.|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#...#|#.#.#|##.##|#...#",
    "X": "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..",
    "Z": "#####|....#|...#.|..#..|.#...|#....|#####",
    "a": ".....|.....|.###.|....#|.####|#...#|.####",
    "b": "#....|#....|#.##.|##..#|#...#|#...#|####.",
    "c": ".....|.....|.###.|#...#|#....|#...#|.###.",
    "d": "....#|....#|.##.#|#..##|#...#|#...#|.####",
    "e": ".....|.....|.###.|#...#|#####|#....|.####",
    "f": "..##|.#..|####|.#..|.#..|.#..|.#..",
    "g": ".....|.....|.####|#...#|#...#|.####|....#|####.",
    "h": "#....|#....|#.##.|##..#|#...#|#...#|#...#",
    "i": "#|.|#|#|#|#|#",
    "j": "...#|....|...#|...#|...#|...#|#..#|.##.",
    "k": "#...|#...|#..#|#.#.|##..|#.#.|#..#",
    "l": "#.|#.|#.|#.|#.|#.|.#",
    "m": ".....|.....|##.#.|#.#.#|#.#.#|#...#|#...#",
    "n": ".....|.....|####.|#...#|#...#|#...#|#...#",
    "o": ".....|.....|.###.|#...#|#...#|#...#|.###.",
    "p": ".....|.....|#.##.|##..#|#...#|####.|#....|#....",
    "q": ".....|.....|.##.#|#..##|#...#|.####|....#|....#",
    "r": ".....|.....|#.##.|##..#|#....|#....|#....",
    "s": ".....|.....|.####|#....|.###.|....#|####.",
    "t": ".#.|.#.|###|.#.|.#.|.#.|..#",
    "u": ".....|.....|#...#|#...#|#...#|#...#|.####",
    "v": ".....|.....|#...#|#...#|#...#|.#.#.|..#..",
    "w": ".....|.....|#...#|#...#|#.#.#|#.#.#|.####",
    "x": ".....|.....|#...#|.#.#.|..#..|.#.#.|#...#",
    "y": ".....|.....|#...#|#...#|#...#|.####|....#|####.",
    "z": ".....|.....|#####|...#.|..#..|.#...|#####",
    "0": ".###.|#...#|#..##|#.#.#|##..#|#...#|.###.",
    "1": "..#..|.##..|..#..|..#..|..#..|..#..|#####",
    "2": ".###.|#...#|....#|..##.|.#...|#....|#####",
    "3": ".###.|#...#|....#|..##.|....#|#...#|.###.",
    "4": "...##|..#.#|.#..#|#...#|#####|....#|....#",
    "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": "..##.|.#...|#....|####.|#...#|#...#|.###.",
    "7": "#####|#...#|....#|...#.|..#..|..#..|..#..",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.",
    "9": ".###.|#...#|#...#|.####|....#|...#.|.##..",
    " ": "...",
    ".": ".|.|.|.|.|.|#",
    ",": ".|.|.|.|.|.|#|#",
    "-": "....|....|....|####",
    "_": ".....|.....|.....|.....|.....|.....|.....|#####",
    "'": "#|#",
    '"': "#.#|#.#",
    "!": "#|#|#|#|#|.|#",
    "?": ".###.|#...#|....#|...#.|..#..|.....|..#..",
    "(": "..#|.#.|#..|#..|#..|.#.|..#",
    ")": "#..|.#.|..#|..#|..#|.#.|#..",
    "&": ".#...|#.#..|.#...|#.#.#|#..#.|#..#.|.##.#",
    "/": "....#|...#.|...#.|..#..|.#...|.#...|#....",
    "+": ".....|..#..|..#..|#####|..#..|..#..",
    ":": ".|.|#|.|.|.|#",
    "#": ".#.#.|.#.#.|#####|.#.#.|#####|.#.#.|.#.#.",
}


def _bitmap(ch: str) -> np.ndarray:
    rows = GLYPHS.get(ch)
    if rows is None:
        plain = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
        rows = GLYPHS.get(plain[:1], GLYPHS["?"])
    lines = rows.split("|")
    width = max(len(r) for r in lines)
    out = np.zeros((ROWS, width), dtype=bool)
    for y, r in enumerate(lines[:ROWS]):
        for x, c in enumerate(r):
            out[y, x] = c == "#"
    return out


def text_mask(text: str) -> np.ndarray:
    """The text in font pixels: True where a pixel is lit."""
    glyphs = [_bitmap(c) for c in text] or [_bitmap(" ")]
    width = sum(g.shape[1] for g in glyphs) + len(glyphs) - 1
    out = np.zeros((ROWS, width), dtype=bool)
    x = 0
    for g in glyphs:
        out[:, x:x + g.shape[1]] = g
        x += g.shape[1] + 1
    return out


def _hex(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def unit_for(video_height: int, text_height: float) -> int:
    """Screen pixels per font pixel, so capital letters are about
    text_height x the video height tall."""
    return max(1, round(text_height * video_height / 7))


def render_tag(text: str, unit: int, opacity: float = 0.45, color: str = "#ffffff",
               font: str = "builtin") -> np.ndarray:
    """An RGBA image (height x width x 4, uint8) of one nametag."""
    if font and font != "builtin" and Path(font).is_file():
        return _render_ttf(text, unit, opacity, color, Path(font))
    mask = text_mask(text)
    h, w = mask.shape[0] + 2 * PAD, mask.shape[1] + 2 * PAD
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[..., 3] = round(255 * opacity)
    ys, xs = np.nonzero(mask)
    img[ys + PAD, xs + PAD, :3] = _hex(color)
    img[ys + PAD, xs + PAD, 3] = 255
    return np.repeat(np.repeat(img, unit, axis=0), unit, axis=1)


def _render_ttf(text: str, unit: int, opacity: float, color: str, path: Path) -> np.ndarray:
    """A tag drawn with your own font file (e.g. a pixel font such as Monocraft)."""
    from PIL import Image, ImageDraw, ImageFont

    size = unit * 9
    font = ImageFont.truetype(str(path), size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    probe.fontmode = "1"
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    asc, desc = font.getmetrics()
    pad = unit * PAD
    w, h = right - left + 2 * pad, asc + desc + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, round(255 * opacity)))
    text_layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(text_layer)
    draw.fontmode = "1"
    draw.text((pad - left, pad), text, font=font, fill=255)
    solid = Image.new("RGBA", (w, h), (*_hex(color), 255))
    img.paste(solid, (0, 0), text_layer)
    return np.asarray(img)


def save_png(rgba: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(path)
