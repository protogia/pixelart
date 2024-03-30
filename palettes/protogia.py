from typing import List
from palettes.default import DefaultPalette

class Palette(DefaultPalette):
    colors = [
    "#2d2d2d", "#393939", "#515151", "#747369",
    "#a09f93", "#d3d0c8", "#e8e6df", "#f2f0ec",
    "#f2777a", "#f99157", "#ffcc66", "#99cc99",
    "#66cccc", "#6699cc", "#cc99cc", "#d27b53"
    ]
    
    def __init__(self):
        super().__init__(self.colors)