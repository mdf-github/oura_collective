"""
Utility functions
"""
from pymer4.models import Lmer, Lm
from pathlib import Path
import pandas as pd


def remove_filepath_ext(filepath, as_path=True):
    filepath = Path(filepath)
    extensions = "".join(filepath.suffixes)
    
    f = str(filepath).removesuffix(extensions)
    if as_path:
        f = Path(f)
    return f

def get_pct_srs(srs, level=None):
    if level is None:
        df = pd.concat([srs.rename('size').to_frame(), (srs / srs.sum()).rename('frac').to_frame()], axis=1)
    else:
        c = srs / srs.groupby(level=level).sum()
        df = pd.concat([srs.rename('size').to_frame(), c.rename('frac').to_frame()], axis=1)
    return df

def savefig(fig, outputfile, dpi=400, bbox_inches='tight', **kwargs):
    if outputfile is not None:
        fig.savefig(outputfile, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(outputfile)

def savefig_multext(fig, outputfile, exts=['.png', '.pdf'], **kwargs):
    base_outputfile = remove_filepath_ext(outputfile)
    for ext in exts:
        savefig(fig, f'{base_outputfile}{ext}', **kwargs)

def add_dayofweek(data, date_col='summary_date', copy=True):
    if copy:
        data = data.copy()

    data[date_col] = pd.to_datetime(data[date_col])
    data['dayofweek'] = data[date_col].dt.dayofweek
    data['day_name'] = data[date_col].dt.day_name()

    return data

def do_lmm(data, formula, family, verbose=False, **fit_kwargs):
    print('running...')
    model = Lmer(data=data, formula=formula, family=family)
    s = model.fit(**fit_kwargs)
    if verbose:
        print(s)
    return model

def do_glm(data, formula, family, **fit_kwargs):
    print('running...')
    model = Lm(formula, data=data, family=family)
    print(model.fit(**fit_kwargs))
    return model

def run_lmm(*args, **kwargs):
    mod = do_lmm(*args, **kwargs)
    
    return mod.anova()

def run_glm(*args, **kwargs):
    mod = do_glm(*args, **kwargs)
    
    return mod