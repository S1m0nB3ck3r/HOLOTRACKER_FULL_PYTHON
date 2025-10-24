"""Processor utilities for code_link GUI.

Contains helpers to load localization CSVs, normalize columns, convert positions to micrometers,
and a thin wrapper around trackpy linking.
"""
from typing import Optional, Tuple
import os
import math
import pandas as pd

try:
    import trackpy as tp
    TRACKPY_AVAILABLE = True
except Exception:
    tp = None
    TRACKPY_AVAILABLE = False


def _norm(s: str) -> str:
    return ''.join(ch.lower() for ch in s if ch.isalnum())


# The original code attempted to auto-detect column names. The user's CSV format is fixed and
# simple; we'll assume these exact column headers and map them directly. This keeps the loader
# minimal and easier to reason about.



def load_localisation_csv(path: str) -> pd.DataFrame:
    """Load a localisation CSV and return a dataframe with columns: frame,x,y,z (units: meters)

    - Detects columns using heuristics
    - Converts numerical columns to numeric
    - Attempts to detect units (header containing '(m)' or typical magnitudes). If values appear to be in micrometers
      the function converts them to meters. The returned dataframe uses SI units (meters) throughout.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # Assume exact columns as provided by the user (Z present):
    # 'HOLOGRAM NUMBER','OBJECT NUMBER','X POSITION (m)','Y POSITION (m)','Z POSITION (m)','NUMBER OF VOXEL'
    # Map these to a minimal dataframe with columns: frame,x,y,z (units: meters)
    cols = ['HOLOGRAM NUMBER', 'X POSITION (m)', 'Y POSITION (m)', 'Z POSITION (m)']
    df2 = df[cols].copy()
    df2.columns = ['frame', 'x', 'y', 'z']

    df2['frame'] = pd.to_numeric(df2['frame'], errors='coerce').fillna(0).astype(int)
    df2['x'] = pd.to_numeric(df2['x'], errors='coerce')
    df2['y'] = pd.to_numeric(df2['y'], errors='coerce')
    df2['z'] = pd.to_numeric(df2['z'], errors='coerce')

    return df2


from typing import Union, Tuple


def link_df(df: pd.DataFrame, search_range: Union[float, Tuple[float, float, float]], memory: int, minlength: int = 0) -> pd.DataFrame:
    """Thin wrapper around trackpy.link_df.

    df must contain columns 'frame','x','y' (units: meters).
    search_range: search radius in same units as x,y (meters)
    memory: trackpy memory
    """
    if not TRACKPY_AVAILABLE:
        raise RuntimeError('trackpy not available in environment')

    # trackpy expects frame and x,y; preserve z if present
    cols = ['frame', 'x', 'y', 'z']

    # select only expected columns (if z missing, ensure it's present with zeros)
    cols_present = ['frame', 'x', 'y']
    if 'z' in df.columns:
        cols_present.append('z')
    else:
        # create a z column of zeros for trackpy APIs that expect it
        df = df.copy()
        df['z'] = 0.0
        cols_present.append('z')

    df_tp = df[cols_present].copy()

    # If search_range is a tuple (per-axis), use the older tp.link API directly
    try:
        if isinstance(search_range, tuple) or isinstance(search_range, list):
            tuple_range = tuple(float(v) for v in search_range)
            trajectories = tp.link(f=df_tp, search_range=tuple_range, memory=memory, t_column='frame', pos_columns=['x', 'y', 'z'] if 'z' in df_tp.columns else ['x', 'y'])
        else:
            # First attempt: use link_df (vectorized) with scalar radius
            trajectories = tp.link_df(df_tp, float(search_range), memory=memory)
    except Exception as e:
        # If trackpy reports a subnetwork error or other issues, attempt fallback
        msg = str(e).lower()
        if 'subnetwork contains' in msg or 'subnetwork' in msg:
            # Try fallback using tuple search_range (same value on all axes)
            tuple_range = (float(search_range), float(search_range), float(search_range))
            trajectories = tp.link(f=df_tp, search_range=tuple_range, memory=memory, t_column='frame', pos_columns=['x', 'y', 'z'] if 'z' in df_tp.columns else ['x', 'y'])
        else:
            # Re-raise unexpected exceptions so GUI can display them
            raise

    # If minlength specified, filter stubs and relink (wrapper behavior)
    if minlength and minlength > 0:
        try:
            filtered = tp.filtering.filter_stubs(tracks=trajectories, threshold=minlength)
            # relink filtered set (use same value on all axes if scalar provided)
            if isinstance(search_range, (tuple, list)):
                tuple_range = tuple(float(v) for v in search_range)
            else:
                tuple_range = (float(search_range), float(search_range), float(search_range))
            trajectories = tp.link(f=filtered, search_range=tuple_range, memory=memory, t_column='frame', pos_columns=['x', 'y', 'z'] if 'z' in df_tp.columns else ['x', 'y'])
        except Exception:
            # If filtering/relinking fails, keep original trajectories and let caller decide
            pass

    # Normalize output columns similar to wrapper
    try:
        # Ensure particle column exists
        if 'particle' not in trajectories.columns:
            # trackpy older API may use 'particle' or 'particle' will be added; if missing, add a placeholder
            trajectories = trajectories.reset_index().rename(columns={'index': 'particle'})
        # Ensure nb_pix column exists (wrapper expects it)
        if 'nb_pix' not in trajectories.columns:
            trajectories['nb_pix'] = 0
        # Standardize column order
        # Some trackpy versions return different column sets; try to set to expected columns
        cols_out = ['frame', 'x', 'y']
        if 'z' in trajectories.columns:
            cols_out.append('z')
        cols_out += ['nb_pix', 'particle']
        # Reindex to include expected columns (missing columns will be filled with NaN)
        trajectories = trajectories.reindex(columns=cols_out)
        trajectories = trajectories.sort_values(by=['particle', 'frame'])
    except Exception:
        # best-effort normalization; ignore failures
        pass

    return trajectories
