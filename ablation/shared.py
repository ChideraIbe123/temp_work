"""Shared data loading and helpers for ablation study."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
EXCEL_FILE = os.path.join(DATA_DIR, 'Data for Datathon (Revised).xlsx')
TEMPLATE_FILE = os.path.join(DATA_DIR, 'template_forecast_v00.csv')
PORTFOLIOS = ['A', 'B', 'C', 'D']

def load_data():
    xlsx = pd.ExcelFile(EXCEL_FILE)
    template = pd.read_csv(TEMPLATE_FILE)

    daily = {}
    for p in PORTFOLIOS:
        df = pd.read_excel(xlsx, f'{p} - Daily')
        df['Date'] = pd.to_datetime(df['Date'].str.strip().str.rsplit(' ', n=1).str[0], format='%m/%d/%y')
        df = df.sort_values('Date').reset_index(drop=True)
        df.columns = [c.strip() for c in df.columns]
        daily[p] = df

    mmap = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
            'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
    intervals = {}
    for p in PORTFOLIOS:
        df = pd.read_excel(xlsx, f'{p} - Interval')
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=['Interval']).copy()
        df['mnum'] = df['Month'].map(mmap)
        df['Day'] = df['Day'].astype(int)
        df['Date'] = pd.to_datetime(dict(year=2024, month=df['mnum'], day=df['Day']))
        df['slot'] = df['Interval'].apply(lambda t: t.hour*2 + t.minute//30)
        df = df.sort_values(['Date','slot']).reset_index(drop=True)
        intervals[p] = df

    staff = pd.read_excel(xlsx, 'Daily Staffing')
    staff.columns = ['Date'] + [f'Staff_{p}' for p in PORTFOLIOS]
    staff['Date'] = pd.to_datetime(staff['Date'])
    staff = staff.sort_values('Date').reset_index(drop=True)

    return daily, intervals, staff, template


HOLIDAYS = pd.to_datetime(['2024-01-01','2024-01-15','2024-02-19','2024-05-27','2024-06-19',
    '2024-07-04','2024-09-02','2024-10-14','2024-11-11','2024-11-28','2024-12-25',
    '2025-01-01','2025-01-20','2025-02-17','2025-05-26','2025-06-19',
    '2025-07-04','2025-09-01','2025-10-13','2025-11-11','2025-11-27','2025-12-25'])


def make_features(df, port, staff):
    f = pd.DataFrame()
    f['Date'] = df['Date']
    f['dow'] = df['Date'].dt.dayofweek
    f['dom'] = df['Date'].dt.day
    f['month'] = df['Date'].dt.month
    f['woy'] = df['Date'].dt.isocalendar().week.astype(int)
    f['year'] = df['Date'].dt.year
    f['wknd'] = (f['dow'] >= 5).astype(int)
    f['mon'] = (f['dow'] == 0).astype(int)
    f['dow_s'] = np.sin(2*np.pi*f['dow']/7)
    f['dow_c'] = np.cos(2*np.pi*f['dow']/7)
    f['month_s'] = np.sin(2*np.pi*f['month']/12)
    f['month_c'] = np.cos(2*np.pi*f['month']/12)
    f['holiday'] = df['Date'].isin(HOLIDAYS).astype(int)
    f['month_start'] = (f['dom'] <= 5).astype(int)
    f['month_end'] = (f['dom'] >= 26).astype(int)
    for m in ['Call Volume','CCT','Abandon Rate']:
        if m not in df.columns: continue
        f[f'{m}_l7'] = df[m].shift(7)
        f[f'{m}_l14'] = df[m].shift(14)
        f[f'{m}_l28'] = df[m].shift(28)
        f[f'{m}_l365'] = df[m].shift(365)
        f[f'{m}_r7'] = df[m].rolling(7).mean()
        f[f'{m}_r30'] = df[m].rolling(30).mean()
        f[f'{m}_ew'] = df[m].ewm(span=7).mean()
    sc = f'Staff_{port}'
    f = f.merge(staff[['Date',sc]].rename(columns={sc:'agents'}), on='Date', how='left')
    for m in ['Call Volume','CCT','Abandon Rate']:
        if m in df.columns:
            f[f'tgt_{m}'] = df[m]
    return f


def build_profiles(intervals, daily):
    cv_prof, abd_prof, cct_prof, prof_cct_avg = {}, {}, {}, {}
    for p in PORTFOLIOS:
        df = intervals[p].copy()
        df['dow'] = df['Date'].dt.dayofweek
        dtot = df.groupby('Date')['Call Volume'].sum().reset_index()
        dtot.columns = ['Date','dtot']
        df = df.merge(dtot, on='Date')
        abt = df.groupby('Date')['Abandoned Calls'].transform('sum')
        df['cv_pct'] = df['Call Volume'] / df['dtot'].replace(0, np.nan)
        df['abd_pct'] = df['Abandoned Calls'] / abt.replace(0, np.nan)
        cv_prof[p], abd_prof[p], cct_prof[p] = {}, {}, {}
        for dow in range(7):
            sub = df[df['dow']==dow]
            for col, store in [('cv_pct',cv_prof), ('abd_pct',abd_prof)]:
                pr = sub.groupby('slot')[col].median()
                a = np.zeros(48)
                a[pr.index.astype(int)] = pr.values
                a = np.nan_to_num(a, 0)
                a = gaussian_filter1d(a, sigma=0.7)
                if a.sum() > 0: a /= a.sum()
                store[p][dow] = a
            pr = sub.groupby('slot')['CCT'].median()
            a = np.zeros(48)
            a[pr.index.astype(int)] = pr.values
            a = np.nan_to_num(a, 0)
            a = gaussian_filter1d(a, sigma=0.7)
            cct_prof[p][dow] = a
        msk = (daily[p]['Date']>='2024-04-01') & (daily[p]['Date']<='2024-06-30')
        prof_cct_avg[p] = daily[p].loc[msk, 'CCT'].mean()
    return cv_prof, abd_prof, cct_prof, prof_cct_avg


def feat_cols(df):
    skip = ['Date','tgt_Call Volume','tgt_CCT','tgt_Abandon Rate']
    return [c for c in df.columns if c not in skip]


def score_submission(res, daily):
    ratios = {}
    for p in PORTFOLIOS:
        a24 = daily[p][(daily[p]['Date'].dt.month==8)&(daily[p]['Date'].dt.year==2024)]
        fc = res[p]['cv'].sum()
        ac = a24['Call Volume'].sum()
        ratios[p] = fc / ac
    abd_rates = {}
    for p in PORTFOLIOS:
        cv = res[p]['cv']
        abd_rates[p] = res[p]['ar'][cv > 0].mean()
    return ratios, abd_rates
