import pandas as pd
import re
from datetime import datetime

def expand_year_ranges_yyyy(text, current_year=None):
    """
    Expand year ranges to include all individual years:
    - [2010-2014] -> [2010, 2011, 2012, 2013, 2014]
    - [2018-] -> [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    - [2018, 2019] -> [2018, 2019] (already expanded)
    """
    if pd.isna(text):
        return text
    
    if current_year is None:
        current_year = datetime.now().year
    
    def replace_range(match):
        full_match = match.group(0)  # e.g., "[2018-]" or "[2010-2014]"
        start_year = match.group(1)  # e.g., "2018" or "2010"
        end_year = match.group(2)    # e.g., "" or "2014"
        
        try:
            start = int(start_year)
            
            # Handle open-ended ranges like [2018-]
            if not end_year:
                end = current_year
            else:
                end = int(end_year)
            
            # Generate all years in the range
            years = list(range(start, end + 1))
            years_str = ', '.join(map(str, years))
            
            return f"[{years_str}]"
        except ValueError:
            # If conversion fails, return original
            return full_match
    
    # Pattern to match [YYYY-YYYY] or [YYYY-]
    pattern = r'(\d{4})-(\d{4}|)'
    result = re.sub(pattern, replace_range, text)
    
    return result

def expand_year_ranges_yy(text, current_year=2025):
    """
    Expand short year ranges like:
    - 14-18 → 2014, 2015, 2016, 2017, 2018
    - 20- → 2020, 2021, 2022, 2023, 2024, 2025
    
    But NOT model codes like:
    - G22-26 (stays as G22-26)
    - G05 (stays as G05)
    """
    if pd.isna(text):
        return text
    
    def replace_range(match):
        start_year = match.group(1)  # e.g., "20"
        end_year = match.group(2)    # e.g., "18" or ""
        
        try:
            # Convert 2-digit years to 4-digit (assuming 2000s for car models)
            start = int(start_year)
            start_full = 2000 + start if start < 50 else 1900 + start
            
            # Handle open-ended ranges like 20-
            if not end_year:
                end_full = current_year
            else:
                end = int(end_year)
                end_full = 2000 + end if end < 50 else 1900 + end
            
            # Generate all years in the range
            years = list(range(start_full, end_full + 1))
            years_str = ', '.join(map(str, years))
            
            return years_str
            
        except ValueError:
            # If conversion fails, return original
            return match.group(0)
    
    # Pattern with negative lookbehind to avoid matching after letters
    # (?<![A-Za-z]) ensures no letter immediately before the year pattern
    pattern = r'(?<![A-Za-z])(\d{2})-(\d{2}|)(?=\)|$|\s)'
    result = re.sub(pattern, replace_range, text)
    
    return result

def expand_vehicle_years(df, mode:str, column:str, current_year=2025):
    """
    Expand year ranges in the vehicle column of the dataframe
    """
    if mode =="full":
        df = df.copy()
        df[column] = df[column].apply(
            lambda x: expand_year_ranges_yyyy(x, current_year)
        )
        return df
    elif mode == "short":
        df = df.copy()
        df[column] = df[column].apply(
            lambda x: expand_year_ranges_yy(x, current_year)
        )
        return df
    else:
        raise Exception('Unknown mode...')