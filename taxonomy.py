import pandas as pd

# ----------------------------------------------------------------------
# Supernovae to avoid using for template creation
# Derived from baseline inspection (dr4_bts_baseline_cutselect)
# or lightcurve matching (S. Hackali)
# ----------------------------------------------------------------------

SN_REJECT = {
    'LATE_PEAK': [
        'ZTF19aatdcnq', 'ZTF19aanneuv',
  'ZTF19abyjnub',  'ZTF19acbveqm',
  'ZTF18acwwljq',  'ZTF20aauonhf',
  'ZTF20aauyhwo',  'ZTF18aaeraoq',
  'ZTF18aawppyt',  'ZTF18aatfxtd',
  'ZTF18aawqjjj',  'ZTF18aaviawx',
  'ZTF18abdnffn',  'ZTF18abkxfou',
  'ZTF18ablwvcy',  'ZTF18acgxwdr',
  'ZTF18acotwcs',  'ZTF18acrxnfr',
  'ZTF18acyyirk',  'ZTF19aadufor',
  'ZTF19aamjjcx',  'ZTF20abidglx',
  'ZTF20abwtifz',  'ZTF23abefecu',
  'ZTF20acxkenf',  'ZTF20acynudw',
  'ZTF21aaocwmx',  'ZTF21acjnvoo',
  'ZTF21acqceqc',  'ZTF21acudpir',
  'ZTF22aaedurh',  'ZTF22aavzsiz',
  'ZTF22ablpkba',  'ZTF22abmszqi',
  'ZTF23abczkxz',  'ZTF23abhpuoc',
  'ZTF23abhvuft',  'ZTF23abiflhi',
  'ZTF23abjrdem',  'ZTF23abpuxjl',
  'ZTF23abqgjjm',  'ZTF23abqygjv',
  'ZTF23absapog',  'ZTF23abscfuu'
    ],
    'MULTIPEAK': [
        'ZTF19aawlgne',
  'ZTF19aazdroc',  'ZTF19aaywash',
  'ZTF19ablkgbs',  'ZTF19ablujxj',
  'ZTF19abtrjqg',  'ZTF19abyjnub',
  'ZTF19abyrano',  'ZTF19abynjou',
  'ZTF19abvjnsm',  'ZTF18aaojsvk',
  'ZTF19acewufb',  'ZTF18actwrnh',
  'ZTF19acwsajh',
  'ZTF20aadzfzy',
  'ZTF18acwwljq',
  'ZTF20aahmtso',
  'ZTF20aaptcqc',
  'ZTF18aaakpbx',
  'ZTF18aafeiiy',
  'ZTF18aaeraoq',
  'ZTF18aaddhpb',
  'ZTF18aaaorhy',
  'ZTF18aagrkwn',
  'ZTF18aahjbii',
  'ZTF18aagteoy',
  'ZTF18aahtkze',
  'ZTF18aakkvea',
  'ZTF18aakvxqj',
  'ZTF18aahxmjf',
  'ZTF18aakqtwt',
  'ZTF18aanxbdh',
  'ZTF18aassbst',
  'ZTF18aanggvq',
  'ZTF18aaraifg',
  'ZTF18aaspova',
  'ZTF18aauuohe',
  'ZTF18aaxnnev',
  'ZTF18aayiqyq',
  'ZTF18abadmpz',
  'ZTF18abcbwnd',
  'ZTF18abcysck',
  'ZTF18abcmsux',
  'ZTF18abeaytd',
  'ZTF18abdgltl',
  'ZTF18aazndjw',
  'ZTF18abjyjfe',
  'ZTF18abkijlr',
  'ZTF18ablookq',
  'ZTF18abnopie',
  'ZTF18ablvazp',
  'ZTF18ablrlbm',
  'ZTF18absmpcp',
  'ZTF18absltuj',
  'ZTF18absewls',
  'ZTF18abtrowl',
  'ZTF18abtlwvs',
  'ZTF18abtnfmp',
  'ZTF18abusrtw',
  'ZTF18abvvyro',
  'ZTF18abvftig',
  'ZTF18abxcues',
  'ZTF18acaisao',
  'ZTF18acajzuh',
  'ZTF18acbwjyx',
  'ZTF18acdvvnc',
  'ZTF18acejvgr',
  'ZTF18aceijsp',
  'ZTF18acfvhko',
  'ZTF18aceaape',
  'ZTF18achbuhn',
  'ZTF18achphjo',
  'ZTF18acketah',
  'ZTF18acusuwi',
  'ZTF18acvgvdz',
  'ZTF18acwrrpt',
  'ZTF18aczyagh',
  'ZTF18adjkgrv',
  'ZTF18adgvgdk',
  'ZTF18adlaloa',
  'ZTF18admhzsq',
  'ZTF19aacipwt',
  'ZTF19aademoc',
  'ZTF19aakiygv',
  'ZTF19aakpkro',
  'ZTF20abavxkq',
  'ZTF20abfcszi',
  'ZTF20abpbfzp',
  'ZTF20abxclnl',
  'ZTF20acpguww',
  'ZTF20acszwan',
  'ZTF23abefecu',
  'ZTF20acxkenf',
  'ZTF20acxffhe',
  'ZTF20acvamwo',
  'ZTF21aadnzyf',
  'ZTF21aagkfhm',
  'ZTF21aaopstx',
  'ZTF21aaqqwsa',
  'ZTF21aatlewl',
  'ZTF21aauarkg',
  'ZTF21aazlmwl',
  'ZTF21abccdld',
  'ZTF21abnfdqg',
  'ZTF21acowqep',
  'ZTF22aadhosm',
  'ZTF22aafxpui',
  'ZTF22aaikall',
  'ZTF22aajkfrp',
  'ZTF22aamndbq',
  'ZTF22aaqwdjw',
  'ZTF22aavzsiz',
  'ZTF22ablnjgv',
  'ZTF22abrpjxu',
  'ZTF22absamkh',
  'ZTF22abugigx',
  'ZTF23aabnwws',
  'ZTF23aadchqd',
  'ZTF23abscfuu'
  ],
 'COINCIDENT': [
     'ZTF18acidjch', 'ZTF23aabzrgo', 'ZTF24aadxdqn', 'ZTF21abnfdqg'
     ],
 'RECURRENT': [
     'ZTF18aacmwsq', 'ZTF18aatfxtd', 'ZTF18adjtukb', 'ZTF18aavoyoq', 'ZTF20acycgkl'
     ],
'HACKALI': ['ZTF18acwyvet', 'ZTF18adbikdz', 'ZTF19aadnxbh', 'ZTF19acxxwrs', 'ZTF20aalrqbu', 
    'ZTF20aamttiw', 'ZTF20aazycgy', 'ZTF20abcjdwu', 'ZTF20abefbpl', 'ZTF20accmutv', 
    'ZTF20aciwcuz', 'ZTF21aabpnlm', 'ZTF21acadrzr', 'ZTF22aaaefgq']
    }






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

