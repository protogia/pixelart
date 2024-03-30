from typing import List
import string

class DefaultPalette:
    colors = []

    def __init__(self, list_of_hexcolors: List = []):
        if self.__validate(list_of_hexcolors):
            self.colors = list_of_hexcolors
        else:
            self.colors = []

    def __validate(self, list_of_hexcolors: List) -> bool:
        for entry in list_of_hexcolors:
            assert type(entry) == str
            assert entry[0] == '#'
            assert all(c in string.hexdigits for c in entry[1:])
        return True
            
    def hex_to_rgb(self):
        colors = []
        for hex_code in self.colors:
            hex_code = hex_code.lstrip('#')
            r, g, b = hex_code[:2], hex_code[2:4], hex_code[4:]
            c = [int(x, 16) for x in (r, g, b)] # Convert each component to decimal (0-255)
            colors.append(c)
        return colors 
