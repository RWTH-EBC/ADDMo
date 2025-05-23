import os
import re
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field, create_model

dark_red = [172 / 255, 43 / 255, 28 / 255]
red = [221 / 255, 64 / 255, 45 / 255]
light_red = [235 / 255, 140 / 255, 129 / 255]
green = [112 / 255, 173 / 255, 71 / 255]
light_grey = [217 / 255, 217 / 255, 217 / 255]
grey = [157 / 255, 158 / 255, 160 / 255]
dark_grey = [78 / 255, 79 / 255, 80 / 255]
light_blue = [157 / 255, 195 / 255, 230 / 255]
blue = [0 / 255, 84 / 255, 159 / 255]
black = [0, 0, 0]
ebc_palette_sort_1 = [dark_red,red,light_red,dark_grey,grey,light_grey,blue,light_blue,green]
ebc_palette_sort_2 = [red,blue,grey,green,dark_red,dark_grey,light_red,light_blue,light_grey]

def cm2inch(value):
    return value / 2.54

def save_pdf(plt,save_path):
    save_path= os.path.join(save_path)+ ".pdf"
    plt.savefig(save_path, dpi=900, bbox_inches="tight", format='pdf')

def sanitize_column_name(col: str) -> str:
    """Convert column name to valid Python identifier"""
    # Replace all special characters with single underscore
    col = re.sub(r'[^a-zA-Z0-9]', '_', col)
    # Collapse multiple underscores to single
    col = re.sub(r'_+', '_', col)
    # Remove leading/trailing underscores
    return col.strip('_')

def create_bounds_model(df: pd.DataFrame) -> type[BaseModel]:
    fields = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            sanitized_col = sanitize_column_name(col)
            fields[f"{sanitized_col}_min"] = (Optional[float], Field(default=df[col].min(), title=f"{col} min"))
            fields[f"{sanitized_col}_max"] = (Optional[float], Field(default=df[col].max(), title=f"{col} max"))
    DynamicBoundsModel = create_model("DynamicBoundsModel", **fields)
    return DynamicBoundsModel
