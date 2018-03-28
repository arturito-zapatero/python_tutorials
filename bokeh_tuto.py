# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 00:03:17 2017

@author: aszewczyk
"""

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap

output_file("periodic.html")

periods = ["I", "II", "III", "IV", "V", "VI", "VII"] #rownames
groups = [str(x) for x in range(1, 19)]              #colnames

#clean data
df = elements.copy()                                 #df with values in long format
df["atomic mass"] = df["atomic mass"].astype(str)    
df["group"] = df["group"].astype(str)
df["period"] = [periods[x-1] for x in df.period]    #change arabic numbers to roman 
df = df[df.group != "-"]
df = df[df.symbol != "Lr"]
df = df[df.symbol != "Lu"]


##color map
cmap = {
    "alkali metal"         : "#FF3333",
    "alkaline earth metal" : "#1f78b4",
    "metal"                : "#d93b43",
    "halogen"              : "#999d9a",
    "metalloid"            : "#e08d49",
    "noble gas"            : "#eaeaea",
    "nonmetal"             : "#f1d4Af",
    "transition metal"     : "#599d7A",
}

source = ColumnDataSource(df) #maps names of columns to sequences or arrays.

#create an empty tamplate for table, groups and periods must be lists of strings
p = figure(title="Periodic Table (omitting LA and AC Series)", plot_width=1000, plot_height=450,
           tools="", 
           toolbar_location=None,
           x_range=groups, y_range=list(reversed(periods)))

#draws rectangles and colors on empty table
p.rect("group", "period", 0.95, 0.95, 
       #maps names of columns to sequences or arrays.
       source=source, 
       fill_alpha=0.6, 
       #plot legend
       legend="metal",
       #define colors using cmap variable and metal column in df (different metals have different field colors)
       color=factor_cmap('metal', palette=list(cmap.values()), factors=list(cmap.keys())))

#add text to each of the fields
text_props = {"source": source, "text_align": "left", "text_baseline": "middle"}

x = dodge("group", -0.4, range=p.x_range)

r = p.text(x=x, y="period", text="symbol", **text_props)
r.glyph.text_font_style="bold"

r = p.text(x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number", **text_props)
r.glyph.text_font_size="8pt"

r = p.text(x=x, y=dodge("period", -0.35, range=p.y_range), text="name", **text_props)
r.glyph.text_font_size="5pt"

r = p.text(x=x, y=dodge("period", -0.2, range=p.y_range), text="atomic mass", **text_props)
r.glyph.text_font_size="5pt"

p.text(x=["3", "3"], y=["VI", "VII"], text=["LA", "AC"], text_align="center", text_baseline="middle")



#add values of each field to be shown after hovering
p.add_tools(HoverTool(tooltips = [
    ("Name", "@name"),
    ("Atomic number", "@{atomic number}"),
    ("Atomic mass", "@{atomic mass}"),
    ("Type", "@metal"),
    ("CPK color", "$color[hex, swatch]:CPK"),
    ("Electronic configuration", "@{electronic configuration}"),
]))

p.outline_line_color = None
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_standoff = 0
p.legend.orientation = "horizontal"
p.legend.location ="top_center"

show(p)