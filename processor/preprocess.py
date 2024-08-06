import pandas as pd
import numpy as np
import re
import math
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype, is_bool_dtype
import scipy
import plotly.graph_objects as go
import seaborn as sn
from scipy.cluster import hierarchy as hc
import streamlit as st
import time
import matplotlib.pyplot as plt


@st.cache_data(show_spinner=False)
def remove_null(df, target_variable):
    df.dropna(subset = [target_variable], inplace=True)
    df.dropna(how='all', inplace=True)

@st.cache_data(show_spinner=False)
def remove_lowcounts(df, rows):
  val =round(rows*0.25)
  t = []
  for label,content in df.items():
    if df[label].count() < val:
      df.drop(columns = label, inplace=True)
      t.append(label)
  df.dropna(how='all', inplace=True)
  return t

@st.cache_data(show_spinner=False)
def remove_highcardinals_string(df):
  t = []
  for label,content in df.items():
    val1 = round(df[label].count()*0.65)
    val2 = round(df[label].count()*0.05)
    if is_string_dtype(content):
      if ((df[label].nunique() > val1) and (df[label].value_counts().max() < val2)):
        df.drop(columns = label, inplace=True)
        t.append(label)
  return t

@st.cache_data(show_spinner=False)
def remove_highcardinals_numeric(df, target_variable):
  t = []
  for label,content in df.items():
    if is_numeric_dtype(content):
      if (df[label].nunique() > (df[label].count() * .995 )):
        if not label == target_variable:
          df.drop(columns = label, inplace=True)
          t.append(label)
  return t

@st.cache_data(show_spinner=False)
def numericalize(df, col, name, max_n_cat):
  if not is_numeric_dtype(col) and ( max_n_cat is None or col.nunique()>max_n_cat):
    df[name] = col.cat.codes+1

@st.cache_data(show_spinner=False)
def fix_missing(df, col, name, na_dict):
  if is_numeric_dtype(col):
    if pd.isnull(col).sum() or (name in na_dict):
      df[name+'_na'] = pd.isnull(col)
      filler = na_dict[name] if name in na_dict else col.median()
      df[name] = col.fillna(filler)
      na_dict[name] = filler
  return na_dict

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_sample(df,n):
  idxs = sorted(np.random.permutation(len(df))[:n])
  return df.iloc[idxs].copy()

@st.cache_data(show_spinner=False)
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, na_dict=None, max_n_cat=None, subset=None):
  if not ignore_flds:
    ignore_flds=[]
  if not skip_flds:
    skip_flds=[]
  if subset:
    df = get_sample(df,subset)
  ignored_flds = df.loc[:, ignore_flds]
  df.drop(ignore_flds, axis=1, inplace=True)
  df = df.copy()
  if y_fld is None:
    y = None
  else:
    if not is_numeric_dtype(df[y_fld]):
      df[y_fld] = df[y_fld].cat.codes
    y = df[y_fld].values
    skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

  if na_dict is None:
    na_dict = {}
  for n,c in df.items():
    na_dict = fix_missing(df, c, n, na_dict)
  for n,c in df.items():
    numericalize(df, c, n, max_n_cat)
  df = pd.get_dummies(df, prefix_sep ='*', dummy_na=True)
  df = pd.concat([ignored_flds, df], axis=1)
  res = [df, y, na_dict]
  return res

@st.cache_data(show_spinner=False)
def train_cats(df):
  for label, content in df.items():
    if is_string_dtype(content):
      df[label] = content.astype('category').cat.as_ordered()

@st.cache_data(show_spinner=False)
def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    #df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

#Working Function
#add_datepart(df_raw,'Date')
@st.cache_data(show_spinner=False)
def get_date(df):
  for label,content in df.items():
    if is_datetime64_any_dtype(content):
      st.write(f'Datetime feature - {label} - converted to multiple separate columns like date, month, year..')
      df[label].fillna(np.nan)
      add_datepart(df,label,time=True)


@st.cache_data(show_spinner=False)
def convert_date(df):
  df = df.apply(lambda col: pd.to_datetime(col, errors='ignore')
              if col.dtypes == object
              else col,
              axis=0)
  return df


@st.cache_data(show_spinner=False)
def spear_C(df):
  if df.shape[0] > 30000:
    df_tmp = get_sample(df,30000)
    sp = np.round(scipy.stats.spearmanr(df_tmp, nan_policy='omit').correlation,4)
  else:
    sp = np.round(scipy.stats.spearmanr(df, nan_policy='omit').correlation,4)
  sp_con = hc.ward(sp)
  dendro = hc.dendrogram(sp_con, labels=df.columns, leaf_rotation=90)
  st.pyplot()
  #plt.show()
  #return sp

@st.cache_data(show_spinner=False)
def pearman_C(df):
  c_matrix = df.corr()
  #corr_matrix[target_variable].sort_values(ascending=False)
  fig,ax= plt.subplots()
  fig.set_size_inches(30,20)
  sn.heatmap(c_matrix, square=True,annot=True)
  st.pyplot()
  return c_matrix

