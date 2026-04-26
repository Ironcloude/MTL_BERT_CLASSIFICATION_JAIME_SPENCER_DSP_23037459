"""
Contains standardised thememing for plotting.
"""

from matplotlib.colors import LinearSegmentedColormap

class Colours:
    LIGHT_BLUE = '#92c5de'
    BLUE = '#4C72B0'
    LIGHT_RED = '#f4a582'
    RED = '#C44E52'
    GREEN = "#5CC44E"
    GREY_LIGHT = 'grey'
    GREY_DARK = 'dimgrey'

POLITICAL_CMAP = LinearSegmentedColormap.from_list(
    "political_sentiment", [Colours.RED, "#F5F5F5", Colours.BLUE])

class Fonts:
    DELTA = {
        'ha': 'center', 
        'va': 'bottom', 
        'fontsize': 7, 
        'color': 'black', 
        'fontweight': 'bold'
    }
    EX_MAIN = {
        'ha': 'right', 
        'va': 'center', 
        'fontweight': 'bold', 
        'fontsize': 9.5
    }
    EX_SUB = {
        'ha': 'right', 
        'va': 'center', 
        'fontweight': 'normal', 
        'fontsize': 8, 
        'color': Colours.GREY_DARK
    }

DISPLAY_NAMES = {
    "EX-1":  "BERT STL (512)",
    "EX-2":  "DeBERTav3 STL (512)",
    "EX-3":  "ModernBERT STL (512)",
    "EX-4":  "ELECTRA STL (512)",
    "EX-5":  "DeBERTa STL (512)",
    "EX-6":  "ModernBERT STL (1024)",
    "EX-7":  "ModernBERT STL (2048)",
    "EX-8":  "ModernBERT FROZEN (1024)",
    "EX-9":  "ModernBERT MTL (λ=1.0, 1024)",
    "EX-10": "ModernBERT MTL (λ=0.75, 1024)",
    "EX-11": "ModernBERT MTL (λ=0.5, 1024)",
    "EX-12": "ModernBERT MTL (λ=0.25, 1024)",
    "EX-13": "ModernBERT MTL (λ=0.0, 1024)",
    "EX-14": "ModernBERT MTL (λ=0.25, 2048)",
}