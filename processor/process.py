import pandas as pd
import numpy as np
import re
import math
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype, is_bool_dtype
from sklearn.model_selection import RandomizedSearchCV
import scipy
import plotly.graph_objects as go
import seaborn as sn
from scipy.cluster import hierarchy as hc
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from treeinterpreter import treeinterpreter as ti
from pdpbox import pdp

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def rf_imp_features(model, df):
  return pd.DataFrame({'features':df.columns, 'imp':model.feature_importances_}).sort_values('imp', ascending=False)

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def plot_fi(fi):
  return fi.plot('features', 'imp', 'barh', figsize=(20,10), color = 'dodgerblue', legend=False)

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def split_vals(a,n):
  return a[:n], a[n:]

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def rmse(x,y):
  return math.sqrt(((x-y)**2).mean())

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_score(model, df_trn, y_trn):
    res = [rmse(model.predict(df_trn), y_trn),
                model.score(df_trn, y_trn)]
    if hasattr(model, 'oob_score_'): res.append(model.oob_score_)
    return res

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_sample(df,n):
  idxs = sorted(np.random.permutation(len(df))[:n])
  return df.iloc[idxs].copy()

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_features(score, oob, trn_cols, trn_rows, df_trn, y_trn, n_estimators1, n_estimators2,max_features, min_samples_split, min_samples_leaf,basetopfeats, basemodel):
	if score > 0.98 or score < .85 or oob < .83:
		grid_param = hyperparameters(trn_cols, trn_rows, n_estimators1, n_estimators2,max_features, min_samples_split, min_samples_leaf)
		RFR = RandomForestRegressor(random_state = 1)
		RFR_Random = RandomizedSearchCV(estimator = RFR, param_distributions = grid_param, n_iter = 30, cv = 3, verbose = 2, random_state=42, n_jobs=-1)
		RFR_Random.fit(df_trn, y_trn)
		best_params = RFR_Random.best_params_
		#st.write(best_params)
		min_samples_leaf = best_params['min_samples_leaf']
		min_samples_split = best_params['min_samples_split']
		n_estimators = best_params['n_estimators']
		max_features = best_params['max_features']
		#st.write(best_params)
		if (len(best_params) == 4):
			#st.write("Picking 4 hyperparameters")
			model = RandomForestRegressor(min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators, max_features=max_features, n_jobs = -1, oob_score=True)
		else:
			#st.write("Picking 3 hyperparameters")
			model = RandomForestRegressor(min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators, n_jobs = -1, oob_score=True)
		model.fit(df_trn, y_trn)
		score_n1 = model.score(df_trn, y_trn)
		oob_n1 = model.oob_score_
		#st.write(score_n1,oob_n1)
		if (oob_n1 > oob):
			topfeats = rf_imp_features(model,df_trn)
			#st.write("Returning new top features based on tuning-4")
			return topfeats, model
		else:
			#st.write("Returning baseline top features-4")
			return basetopfeats, basemodel
	else:
		#st.write("Baseline is good. Go for feature importance")
    #basetopfeats = rf_imp_features(model,df_trn)
		return basetopfeats, basemodel

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def hyperparameters(cols,rows, n_estimators1, n_estimators2,max_features, min_samples_split, min_samples_leaf):
  if (cols < 10) and (rows < 20000):
    grid_param = {'n_estimators': n_estimators2,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    #st.write("1")
    return grid_param
  elif (cols < 10) and (rows > 20000):
    grid_param = {'n_estimators': n_estimators1,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    #st.write("2")
    return grid_param
  elif (cols > 10) and (rows < 20000):
    grid_param = {'n_estimators': n_estimators2,
                  'max_features': max_features,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    #st.write("3")
    return grid_param
  else:
    grid_param = {'n_estimators': n_estimators1,
                  'max_features': max_features,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    #st.write("4")
    return grid_param

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def qcut_df(df, df_trn, feats, target_variable, tmean):
  if len(df[feats]) > 20000:
    df['series'] = pd.qcut(df[feats], duplicates='drop', q=10, precision=2)
  else:
    df['series'] = pd.qcut(df[feats], duplicates='drop', q=5, precision=2)
  tmp = df.groupby(['series'])[target_variable].mean()
  maxl_slice = tmp.idxmax().left
  maxr_slice = tmp.idxmax().right
  minl_slice = tmp.idxmin().left
  minr_slice = tmp.idxmin().right
  #st.write(maxl_slice,maxr_slice,minl_slice,minr_slice)
  #st.write(f"From '{feats}' Max category between {maxl_slice} and {maxr_slice} has a mean '{target_variable}' of {round(tmp.max(),4)}")
  st.markdown(f"**Bucketing '{feats}' into multiple categories to give a different perception based on categorization**")
  st.write(f"* From '{feats}' Max category between {maxl_slice} and {maxr_slice} has a mean '{target_variable}' of {round(tmp.max(),2)} thats {round((tmp.max()/tmean)*100,2)}% above overall average {tmean}")
  #st.sidebar.write(f"{feats}")
  #st.sidebar.write(f"b/w {maxl_slice} and {maxr_slice} has mean '{target_variable}' of {round(tmp.max(),4)} thats  ")
  #st.write(f"From '{feats}' Min category between {minl_slice} and {minr_slice} has a mean '{target_variable}' of {round(tmp.min(),4)}")
  st.write(f"* From '{feats}' Min category between {minl_slice} and {minr_slice} has a mean '{target_variable}' of {round(tmp.min(),2)} thats {round((tmp.min()/tmean)*100,2)}% below overall average {tmean}")
  #maxslice = df[(df[feats] > maxl_slice) & (df[feats] < maxr_slice)]
  #minslice = df[(df[feats] > minl_slice) & (df[feats] < minr_slice)]
  #tmp = pd.DataFrame(tmp)
  #tmp = tmp.to_json()
  maxslice = df_trn[(df_trn[feats] >= maxl_slice) & (df_trn[feats] <= maxr_slice)]
  minslice = df_trn[(df_trn[feats] >= minl_slice) & (df_trn[feats] <= minr_slice)]
  return tmp, maxslice, minslice

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def display_visuals(df, df_trn, feats,imp,attr, target_variable, tmean):
  if is_bool_dtype(df[feats]):
    #would have been 1 hot encoded
    bar_plot(df,feats, target_variable)
    maxslice = imp
    minslice = attr
    return maxslice,minslice
    display_texts(feats,imp,attr, target_variable, tmean)
  elif is_numeric_dtype(df[feats]):
    if df[feats].nunique() < 70:
      display_texts(feats,imp,attr, target_variable,tmean)
      bar_plot(df,feats, target_variable)
    else:
      display_texts(feats,imp,attr,target_variable,tmean)
      scatter_plot(df,feats, target_variable)
    df_cut,maxslice,minslice = qcut_df(df, df_trn, feats, target_variable, tmean)
    #tmp = df_cut.groupby([feats])[target_variable].mean()
    st.write(f'#### Bucketed "{feats}" across multiple categories Vs Average "{target_variable}"')
    bar_ploty(df_cut, feats, target_variable)
    #sn_bar_plot(df_cut)
    #display_texts(feats,imp,attr)
    return maxslice,minslice
    #display_texts(feats,imp,attr)
  elif df[feats].nunique() < 10:
    #1-hot encoded
    bar_plot(df,feats, target_variable)
    minattr = df[feats].value_counts(ascending=True).index.tolist()[0]
    maxslice,minslice = cat_slice(df_trn,feats,attr,minattr)
    display_texts(feats,imp,attr,target_variable, tmean)
    return maxslice,minslice
  else:
    #regular category
    if df[feats].nunique() < 70:
      bar_plot(df,feats, target_variable)
    else:
      scatter_plot(df,feats, target_variable)
    minattr = df_trn[feats].value_counts(ascending=True).index.tolist()[0]
    maxattr = df_trn[feats].value_counts(ascending=False).index.tolist()[0]
    maxslice = df_trn[df_trn[feats]==maxattr]
    minslice = df_trn[df_trn[feats]==minattr]
    display_texts(feats,imp,attr,target_variable, tmean)
    return maxslice,minslice
    #display_texts(feats,imp,attr)

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def display_texts(feats,imp,attribute, target_variable, tmean):
  if not attribute:
    st.markdown(f'**Few details about {feats}**')
    st.write(f'"* {feats}" is an important contributor: {imp}% towards Target: "{target_variable}"')
    st.write(f'* Overall Average "{target_variable}" is {tmean}')
  else:
    st.markdown(f'**Few details about {feats}**')
    st.write(f'* From "{feats}" Category: "{attribute}" is the important contributor: {imp}% towards Target: "{target_variable}"')
    st.write(f'* Overall Average "{target_variable}" is {tmean}')

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def cat_slice(df_trn,feats,attr, minattr):
  #merge feats*attr - then - from df_trn filter the column ==1, return df_trn
  maxslice = df_trn[df_trn[feats+'*'+attr] > 0]
  minslice = df_trn[df_trn[feats+'*'+minattr] > 0]
  return maxslice,minslice

#continuous
# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def scatter_plot(df, feats,target_variable ):
	df_tmp = df.groupby([feats])[target_variable].mean().reset_index()
	fig = px.scatter(df_tmp, x=feats, y=target_variable, width=1000,
              hover_data=[feats, target_variable], color=target_variable,
              labels={'pop':feats + 'vs' + target_variable}, color_continuous_scale='Viridis')
	fig.update_layout(legend_orientation="h", plot_bgcolor='rgb(255,255,255)')
	fig.update_xaxes(showgrid=False, zeroline=False)
	fig.update_yaxes(showgrid=False, zeroline=False)
	st.write(fig)

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def bar_ploty(df,feats, target_variable):
  ax = df.plot.bar(x=df.index, y=df.values, figsize=(28,10), color='mediumseagreen',rot=90)
  ax.set_xlabel(feats)
  ax.set_ylabel(target_variable)
  st.pyplot()

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def sn_bar_plot(df):
  ax = sn.barplot(x=df.index, y=df.values, palette='ch:.25')
  ax.set(xlabel='xx', ylabel = 'yy')
  st.pyplot()
  #plt.show()
  #plot.xlabel = feats1
  #plot.ylabel = target_variable

#categorical
# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def bar_plot(df,feats, target_variable):
  df_tmp = df.groupby([feats])[target_variable].mean().reset_index()
  #st.write(type(df_tmp))
  #st.write(df_tmp)
  fig = px.bar(df_tmp, height=450, width=1000, x=feats, y=target_variable,hover_data=[feats, target_variable], color=target_variable, color_continuous_scale='Viridis', barmode ='group',labels={'pop':feats + 'vs'+ target_variable},)
  fig.update_layout(legend_orientation="h", plot_bgcolor='rgb(200,200,200)')
  fig.update_xaxes(showgrid=False, zeroline=False)
  fig.update_yaxes(showgrid=False, zeroline=False)
  st.write(fig)

#@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_contributions(slice, model,df_trn ):
  if len(slice) < 10:
    sample = slice.copy()
  else:
    sample = get_sample(slice,round(len(slice)*.10))
  prediction, bias, contributions = ti.predict(model, sample)
  bias = bias[0]
  prediction = np.mean(prediction)
  result_list = []
  for i in range(len(sample)):
    idxs = np.argsort(contributions[i])
    for c, value, feature in sorted(zip(contributions[i][idxs], sample.iloc[i][idxs], df_trn.columns[idxs])):
      result = (feature, c, value)
      result_list.append(result)
  df_res = pd.DataFrame(result_list, columns=['feature','cont','value'])
  df_tree = df_res.groupby(['feature'])['cont'].agg('mean').reset_index()
  t_conts = df_tree.sort_values(by='cont',ascending=False)[:10]
  l_conts = df_tree.sort_values(by='cont',ascending=True)[:10]
  #Predictions = bias +
  return bias,prediction,df_tree,t_conts,l_conts,sample

# @st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def plot_pdp(model, x, feat_name, clusters=None):
    #feat_name = feat_name or feat
    p = pdp.pdp_isolate(model, x, feature=feat_name, model_features=x.columns)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                        cluster=clusters is not None,
                        n_cluster_centers=clusters)

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def topfeat_drop(features):
  if (features.iloc[0]['imp']) > .85:
    st.write(features.iloc[0]['features'])
    features, model = get_features()
    return features
  else:
    return features


#####################################################################