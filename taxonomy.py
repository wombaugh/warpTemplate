import pandas as pd

# ----------------------------------------------------------------------
# Known classes
# ----------------------------------------------------------------------

KNOWN_CLASSES = [
    "SN IIn-pec",
    "SN II",
    "SN IIb",
    "SN Ia-CSM",
    "SN Ic-pec",
    "SN Ia-SC",
    "SN Iax",
    "SN Ia-pec",
    "SN IIP",
    "SN Icn",
    "SN Ib",
    "SN Ia",
    "SN Ia-91bg",
    "SN Ib-pec",
    "SN II-pec",
    "SLSN-I",
    "SN Ic-BL",
    "SN IIn",
    "SN Ic",
    "SN IIL",
    "SN Ib/c",
    "SLSN-II",
    "SN Ia-91T",
    "SN Ibn",
]

# ----------------------------------------------------------------------
# WARP mappings
# ----------------------------------------------------------------------

WARP_MAP_NARROW = {
    "SN II-pec": "SN II",
    "SN Ib-pec": "SN Ib",
    "SN Icn": "SN Ic",
    "SN IIL": "SN II",
    "SN IIn-pec": "SN IIn",
    "SN Ic-pec": "SN Ic",
}

WARP_MAP_EXTENDED = {
    "SN Ia-91bg": "SN Ia-91bg (e)",
    "SN IIn": "SN IIn (e)",
    "SN IIb": "SN Ib/c (e)",
    "SN Ia-CSM": "SN Ia-pec (e)",
    "SN Ibn": "SN Ibn (e)",
    "SN Ia-SC": "SN Ia-pec (e)",
    "SN Ib": "SN Ib/c (e)",
    "SLSN-II": "SLSN (e)",
    "SN Iax": "SN Ia-pec (e)",
    "SN Ia-91T": "SN Ia-91T (e)",
    "SLSN-I": "SLSN (e)",
    "SN Ic": "SN Ib/c (e)",
    "SN Ia-pec": "SN Ia-pec (e)",
    "SN IIP": "SN II (e)",
    "SN Ic-BL": "SN Ib/c (e)",
    "SN Ia": "SN Ia (e)",
    "SN II": "SN II (e)",
    "SN Ib/c": "SN Ib/c (e)",
}

WARP_MAP_WIDE = {
    "SN II (e)": "SN II (w)",
    "SN Ib (e)": "SN Ib/c (w)",
    "SN Ibn (e)": "SN Ib/c (w)",
    "SN Ia-91T (e)": "SN Ia (w)",
    "SLSN (e)": "SLSN (w)",
    "SN IIn (e)": "SLSN (w)",
    "SN Ia-pec (e)": "SN Ia-pec (w)",
    "SN Ia-91bg (e)": "SN Ia-91bg (w)",
    "SN Ic (e)": "SN Ib/c (w)",
    "SN Ia (e)": "SN Ia (w)",
    "SN Ib/c (e)": "SN Ib/c (w)",
}

WARP_MAP_ALL = {
    "SN Ia (w)": "SN Ia (a)",
    "SLSN (w)": "SN CC (a)",
    "SN Ib/c (w)": "SN CC (a)",
    "SN Ia-pec (w)": "SN Ia (a)",
    "SN II (w)": "SN CC (a)",
    "SN Ia-91bg (w)": "SN Ia (a)",
}

def add_warpclasses(
    df: pd.DataFrame,
    classcolumn: str = "type",
    purge: bool = False,
) -> pd.DataFrame:
    """
    Add hierarchical WARP classifications.

    Creates the columns:

        <classcolumn>_n  : narrow
        <classcolumn>_e  : extended
        <classcolumn>_w  : wide
        <classcolumn>_a  : all

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    classcolumn : str, default='type'
        Column containing SN classifications.

    purge : bool, default=False
        Remove rows whose classes are not in KNOWN_CLASSES.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with WARP classification columns added.
    """
    if classcolumn not in df.columns:
        raise KeyError(f"Column '{classcolumn}' not found.")

    result = df.copy()

    if purge:
        result = result.loc[
            result[classcolumn].isin(KNOWN_CLASSES)
        ].copy()

    result[f"{classcolumn}_n"] = (
        result[classcolumn].replace(WARP_MAP_NARROW)
    )

    result[f"{classcolumn}_e"] = (
        result[f"{classcolumn}_n"].replace(WARP_MAP_EXTENDED)
    )

    result[f"{classcolumn}_w"] = (
        result[f"{classcolumn}_e"].replace(WARP_MAP_WIDE)
    )

    result[f"{classcolumn}_a"] = (
        result[f"{classcolumn}_w"].replace(WARP_MAP_ALL)
    )

    return result



# ----------------------------------------------------------------------
# Which (sncosmo ) templates are close to which WARP classes
# ----------------------------------------------------------------------

TEMPLATE_CLOSE_TYPES = {
    'PopIII' : ['SLSN (w)', 'SLSN (e)', 'SLSN-II', 'SLSN-I'],
    'SN II' : [
        'SN II', 
        "SN II (e)", 
        'SN II (w)', 'SN CC (a)'
    ],
    'SN II-pec': ['SN II (w)', 'SN CC (a)', "SN II (e)",],
    'SN IIL': [
        'SN II', 
        "SN II (e)",
        'SN II (w)', 'SN CC (a)'
    ],
    'SN IIL/P': [
        'SN II', "SN II (e)",
        'SN II (w)', 'SN CC (a)'
    ],
    'SN IIP': [
        'SN IIP', "SN II (e)",
        'SN II (w)', 'SN CC (a)'
    ],
    'SN IIb': [
        "SN Ib/c (e)",
         'SN Ib/c (w)', 'SN CC (a)'
    ],
    'SN IIn': [
        'SN IIn', 'SLSN (w)', "SLSN (e)", 'SLSN-II', 'SLSN-I', # SLSN ...  
        'SN CC (a)',
    ],
    'SN Ia': [
        'SN Ia', 'Ia-pec', 'Ia-CSM', 'Ia-SC', 'SN Ia-SC', 'Ia-91T',   'SN Ia-pec',     # Adding all the Ia subclasses since few other templates
        "SN Ia-91bg (e)", "SN Ia (e)", "SN Ia-91T (e)", 'SN Ia-CSM', 'SN Ia-91bg', 'SN Ia-91T', "SN Iax",
        'SN Ia (w)', 'SN Ia (a)', "SN Ia-91T (w)", 'SN Ia-91bg (w)', 'SN Ia-pec (w)',
    ],
    'SN Ib': [
        'SN Ib', 'SN Ibn', "SN Ib (e)", "SN IIb", 'SN Ib/c', 
        'SN Ib/c (w)', 'SN CC (a)'
    ],
    'SN Ib/c': [
        'SN Ib/c (w)', 'SN CC (a)', "SN Ib/c (e)",  'SN Ib/c',
    ],
    'SN Ic': [
        'SN Ic',  "SN Ic (e)", 'SN Ic-BL', 
        'SN Ib/c (w)', 'SN CC (a)',  'SN Ib/c',
    ],
    'SN Ic-BL': [
        "SN Ic (e)", 'SN Ic-BL',   'SN Ib/c',
        'SN Ib/c (w)', 'SN CC (a)',
    ],
}

