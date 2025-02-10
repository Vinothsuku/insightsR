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
import time
import matplotlib.pyplot as plt
import plotly.express as px
from treeinterpreter import treeinterpreter as ti
from pdpbox import pdp
from processor.preprocess import *
from processor.process import *

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def set_page_title(title):
    st.sidebar.markdown(unsafe_allow_html=True, body=f"""
        <iframe height=0 srcdoc="<script>
            const title = window.parent.document.querySelector('title') \
                
            const oldObserver = window.parent.titleObserver
            if (oldObserver) {{
                oldObserver.disconnect()
            }} \

            const newObserver = new MutationObserver(function(mutations) {{
                const target = mutations[0].target
                if (target.text !== '{title}') {{
                    target.text = '{title}'
                }}
            }}) \

            newObserver.observe(title, {{ childList: true }})
            window.parent.titleObserver = newObserver \

            title.text = '{title}'
        </script>" />
    """)

st.set_page_config(page_title="insightsR", page_icon=":shark:", layout="wide", initial_sidebar_state="expanded")

set_page_title("insightsR")

local_css("style.css")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# df_raw = pd.read_csv(f'{path}{filename}', low_memory=False)
prodheader = "<div><span class='fontblue'><span class='bold'>insightsR</span></div>"
st.sidebar.markdown(prodheader, unsafe_allow_html=True)

st.sidebar.header('1. User Inputs')
df = st.sidebar.file_uploader('Upload csv', type='csv')

if df is not None:
    df_raw = pd.read_csv(df, low_memory=False)
    target_variable = st.sidebar.selectbox('select Target column (numeric) for insights',(df_raw.columns))
    ready = st.sidebar.button("Generate Insights")
    st.sidebar.header('2. Key Insights')
    st.markdown("<div><span class='fontgreen'><span class='bold'>About insightsR</span></div>", unsafe_allow_html=True)
    with st.expander("Short Note", expanded=True):
        st.write("**insightsR provides automated insights for tabular data**. Insights will be statistically generated based on the target column (dependent variable) selected by the user. Currently insightsR is limited to selecting dependent variable that has continuous values (Regression) - for eg) total sales, price, cost, income, salary etc.")
        st.write("**User is only expected to provide a dataset in csv format, select a numeric Target column and click 'Generate Insights' button** in the left pane. Rest is taken care by the tool")
    with st.expander("Insights provided and how to interpret", expanded=True):
        st.markdown("Once the user uploads the data to be analysed and provides the target column against which insights to be provided, insightsR tool will handle **pre-processing of data** which includes removal of null columns, highly correlated columns, columns that has more than 75% null values, identifies and  converts date time columns to multiple columns that helps in providing better insights. This pre-processing step is covered in <span class='fontcoral'><b>Section 2.1</b></span>", unsafe_allow_html=True)
        st.write("After pre-processing, insightsR tool will statistically analyse the dataset using machine learning algorithms and provides insights below:-")
        st.markdown(f"* Provides **top contributing features against the target data column**. For eg) if target is 'Sales' features that might turn important is price, seasonality etc.  insightsR tool provides such highly important features and % of contribution. This is covered in <span class='fontcoral'><b>Section 2.2</b></span>", unsafe_allow_html=True)
        st.markdown(f"* Provides **data visualizations for top 5 contributing features**. insightsR tool puts numeric features into multiple buckets to provide a different perception by aligning data into bucketed categories. This is covered in <span class='fontcoral'><b>Section 2.3</b></span>", unsafe_allow_html=True)
        st.markdown(f"* In <span class='fontcoral'><b>Section 2.4</b></span>, deeper insights are provided based on identified top contributors. Based on a sample picked from the top contributors, insightsR first provides **info on whether the contribution is +ve or -ve and by what %**. Features that brought down/up the mean target value is an immensely useful insight. Further within the sample how did each feature contribute is detailed.", unsafe_allow_html=True)
        st.markdown(f"* In <span class='fontcoral'><b>Section 2.5</b></span>, tool **simulates a scenario:** what happens when the contributor data stays constant and every other column data remains as is - will the target value improves/degrades? This data will provide us view on whether the contributing features has really impacted the target value or is it just part of larger multiple contributing features combined.", unsafe_allow_html=True)
        st.write(f"Based on the insights provided, user could identify areas that can be improved, devise a plan on what would happen if a change is simulated, devise a plan to achieve the target.")

    if ready:
        header1 = "<div><span class='fontgreen'><span class='bold'>2.1 Data snapshot</span></div>"
        st.markdown(header1, unsafe_allow_html=True)
        rows = df_raw.shape[0]; cols = df_raw.shape[1]
        with st.expander("Few stats reg the dataset"):
            st.write(f"Number of records: {rows}, Number of Features/Columns: {cols}")
            st.markdown('Statistics on the dataset across features/columns')
            st.write(df_raw.describe(include='all').T)
            #st.markdown('Number of unique values across the features/columns')
            #st.dataframe(df_raw.T.apply(lambda x: x.nunique(), axis=1))

        df_raw = convert_date(df_raw)

        with st.expander("Highly Correlated Cols"):
            st.write("Columns that are highly correlated with each other are depicted in the below figure (light shaded columns)")
            st.write("Such columns will be suggested for user if they can be removed in order to provide a better insight that are not displaying redundant columns")
            c_matrix = pearman_C(df_raw)
            #st.dataframe(c_matrix.style.highlight_max(axis=0))
            #st.dataframe(c_matrix)

        remove_null(df_raw, target_variable)

        time.sleep(8)

        tmean = round(df_raw[target_variable].mean(),2)

        #st.markdown("### Removing high cardinal columns and Null records")
        #with st.expander("Added/Removed Columns that aids in generating better insights"):
        printtmp1 = remove_lowcounts(df_raw, rows)
        printtmp2 = remove_highcardinals_string(df_raw)
        printtmp3 = remove_highcardinals_numeric(df_raw, target_variable)

        with st.expander("Removed high cardinal columns and Null records"):
            st.markdown(f"Removed columns: {printtmp1}")
            st.markdown(f"Removed columns: {printtmp2}")
            st.markdown(f"Removed columns: {printtmp3}")

        train_cats(df_raw)

        with st.expander("Date Columns being converted"):
            get_date(df_raw)

        with st.expander("Display sample records from the pre-processed dataset"):
            st.write(df_raw.astype('object').head())

        df_trn,y_trn,nas = proc_df(df_raw,target_variable, max_n_cat=10)
        trn_rows = df_trn.shape[0]; trn_cols = df_trn.shape[1]

        basemodel = RandomForestRegressor(n_estimators = 30, oob_score=True, n_jobs=-1)
        basemodel.fit(df_trn, y_trn)
        result = get_score(basemodel, df_trn, y_trn)
        p_rmse = result[0]
        score = result[1]
        oob = result[2]

        #Important
        #st.markdown("### Metrics from the model")
        #st.write(p_rmse,score, oob)

        basetopfeats = rf_imp_features(basemodel,df_trn)

        # basemodel,basetopfeats = feat_imp_check(basetopfeats)

        n_estimators1 = [50, 100]
        n_estimators2 = [50, 100]
        #max_depth = [100, 200]
        max_features = ['log2','sqrt']
        min_samples_split = [3, 5]
        min_samples_leaf = [2, 3]

        # Commented out IPython magic to ensure Python compatibility.
        features, model = get_features(score, oob, trn_cols, trn_rows,df_trn, y_trn, n_estimators1, n_estimators2,max_features, min_samples_split, min_samples_leaf,basetopfeats, basemodel )

        topfeat5 = features[:5]
        topfeat5['Percent'] = topfeat5['imp']*100
        topfeat5n = topfeat5[['features','Percent']]
        topfeat5n.set_index('features', inplace=True)
        st.sidebar.markdown(f"<div><span class='teal'>Important contributors to {target_variable}</span></div>", unsafe_allow_html=True)
        st.sidebar.dataframe(topfeat5n)
        #st.sidebar.write(f"Top 5 contributing features: {topfeat5}")

        #st.bar_chart(features['features'][:10], features['imp'][:10])
        #st.header("2.2 Feature Importance")
        st.write("")
        st.write("")

        st.markdown(f"<div><span class='fontgreen'><span class='bold'>2.2 Data Columns that are Important contributors towards {target_variable}</span></div>", unsafe_allow_html=True)

        plot_fi(features[:10])
        st.pyplot()

        with st.expander("Additional Details"):
            #st.markdown("### Display few records from the dataset")
            st.write(f"Above graph depicts important data columns that were contributing to the target: {target_variable}")
            st.write(f"Details relating how were those features were impacting the {target_variable}, tontinued below..")

        nfeatures = features.copy()
        new = nfeatures["features"].str.split("*", n = 1, expand = True)
        nfeatures["features"] = new[0]
        if (new.shape[1] == 1):
          nfeatures['attribute']=""
        else:
          nfeatures['attribute']=new[1]

        #nfeatures

        feats1, imp1, attr1 = nfeatures['features'].iloc[0], (round((nfeatures['imp'].iloc[0])*100, 2)), nfeatures['attribute'].iloc[0]
        feats2, imp2, attr2 = nfeatures['features'].iloc[1], (round((nfeatures['imp'].iloc[1])*100, 2)), nfeatures['attribute'].iloc[1]
        feats3, imp3, attr3 = nfeatures['features'].iloc[2], (round((nfeatures['imp'].iloc[2])*100, 2)), nfeatures['attribute'].iloc[2]
        feats4, imp4, attr4 = nfeatures['features'].iloc[3], (round((nfeatures['imp'].iloc[3])*100, 2)), nfeatures['attribute'].iloc[3]
        feats5, imp5, attr5 = nfeatures['features'].iloc[4], (round((nfeatures['imp'].iloc[4])*100, 2)), nfeatures['attribute'].iloc[4]

        topfeatures = [features['features'].iloc[0],features['features'].iloc[1],features['features'].iloc[2],features['features'].iloc[3],features['features'].iloc[4],features['features'].iloc[5],features['features'].iloc[6]]

        st.write("")
        st.write("")
        #st.header("2.3 Visualizations - Top 5 Important Contributing Features Vs Target")
        st.markdown("<div><span class='fontgreen'><span class='bold'>2.3 Visualizations - Top 5 Important Features Vs Target</span></div>", unsafe_allow_html=True)

        with st.expander("Top 1st Important Contributor"):
            maxslice1, minslice1 = display_visuals(df_raw,df_trn,feats1,imp1,attr1, target_variable, tmean)

        # df_maxslice = df_raw[(df_raw[feats1] >= 6.16) & (df_raw[feats1] <= 15.0)]

        # maxslice2, minslice2 = display_visuals(df_maxslice,df_trn,feats2,imp2,attr2)
        with st.expander("Top 2nd Important Contributor"):
            maxslice2, minslice2 = display_visuals(df_raw,df_trn,feats2,imp2,attr2, target_variable, tmean)

        with st.expander("Top 3rd Important Contributor"):
            maxslice3, minslice3 = display_visuals(df_raw,df_trn,feats3,imp3,attr3, target_variable, tmean)

        with st.expander("Top 4th Important Contributor"):
            maxslice4, minslice4 = display_visuals(df_raw,df_trn,feats4,imp4,attr4, target_variable, tmean)

        with st.expander("Top 5th Important Contributor"):
            maxslice5, minslice5 = display_visuals(df_raw,df_trn,feats5,imp5,attr5, target_variable, tmean)

        # %time maxslice4, minslice4 = display_visuals(df_raw,df_trn,feats4,imp4,attr4)

        # %time maxslice5, minslice5 = display_visuals(df_raw,df_trn,feats5,imp5,attr5)

        def print_contribs(bias,prediction,feats):
          #st.write("Prediction = Global Mean Contribution + Sum of all contributions"){round(tmp.max(),2)}
          st.write(f"Global Mean Contribution for {target_variable} is {round(bias,2)}")
          st.write(f"Mean Prediction from this sample is {round(prediction,2)}")
          s_preds = prediction - bias
          if s_preds > 0:
              st.markdown(f"<div><span class='green'><span class='bold'>Contribution from this sample is {round(s_preds,2)}</span></div>", unsafe_allow_html=True)
          else:
              st.markdown(f"<div><span class='fontred'><span class='bold'>Contribution from this sample is {round(s_preds,2)}</span></div>", unsafe_allow_html=True)
          #with st.expander("Display all contributions from various features for the MAX sample"):
          #st.write(f"Following are contributions from various features for the MAX sample - \"{feats}\"")
          #st.write(df_tree)
          col1, col2 = st.columns(2)
          with col1:
            st.write(f"Top 10 +ve feature contributions:")
            st.write(t_conts)
          with col2:
            st.write(f"Top 10 -ve feature contributions:")
            st.write(l_conts)

        # pd.option_context('display.max_rows', 500,'display.max_columns', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', None)

        st.write("")
        st.write("")
        #st.header("2.4 Feature contributions to the Target - Indepth view")
        st.markdown("<div><span class='fontgreen'><span class='bold'>2.4 Feature contributions to the Target</span></div>", unsafe_allow_html=True)

        bias,prediction,df_tree,t_conts,l_conts,sample = get_contributions(maxslice1, model,df_trn)
        mstmp = sample[topfeatures].describe(include ='all').T
        st.markdown(f"<div><span class='fontgrey'>1. Max category from 1st important contributor: {feats1} </span></div>", unsafe_allow_html=True)
        with st.expander("About this data sample"):
            st.write(mstmp)

        with st.expander("Details of Indv contributions towards the mean prediction target"):
            print_contribs(bias,prediction,feats1)

            fig = go.Figure(go.Waterfall(
                name = "chart", orientation = "h",
                y = df_tree['feature'],
                x = df_tree['cont'],
                connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}}
            ))
            title = f"Feature Contributions related to MAX sample - \"{feats1}\""
            fig.update_layout(title = title, height=500, width=1000)
            st.write(fig)

        #get_contributions(minslice1)

        bias,prediction,df_tree,t_conts,l_conts,sample = get_contributions(maxslice2, model,df_trn)
        mstmp = sample[topfeatures].describe(include ='all').T
        #st.write('\033[95m' + "*"*100 + '\033[0m')
        st.markdown(f"<div><span class='fontgrey'>2. Max category from 2nd important contributor: {feats2}</span></div>", unsafe_allow_html=True)
        with st.expander("About this data sample"):
            st.write(mstmp)

        with st.expander("Details of Indv contributions towards the mean prediction target"):
            print_contribs(bias,prediction,feats2)

            fig = go.Figure(go.Waterfall(
                name = "chart", orientation = "h",
                y = df_tree['feature'],
                x = df_tree['cont'],
                connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}}
            ))
            title = f"Feature Contributions related to MAX sample - \"{feats2}\""
            fig.update_layout(title = title, height=500, width=1000)
            st.write(fig)


        # Commented out IPython magic to ensure Python compatibility.
        bias,prediction,df_tree,t_conts,l_conts,sample = get_contributions(maxslice3, model,df_trn)
        mstmp = sample[topfeatures].describe(include ='all').T
        st.markdown(f"<div><span class='fontgrey'>3. Max category from 3rd important contributor: {feats3}</span></div>", unsafe_allow_html=True)
        with st.expander("About this data sample"):
            st.write(mstmp)

        with st.expander("Details of Indv contributions towards the mean prediction target"):
            print_contribs(bias,prediction,feats3)

            fig = go.Figure(go.Waterfall(
                name = "chart", orientation = "h",
                y = df_tree['feature'],
                x = df_tree['cont'],
                connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}}
            ))
            title = f"Feature Contributions related to MAX sample - \"{feats3}\""
            fig.update_layout(title = title, height=500, width=1000)
            st.write(fig)

        bias,prediction,df_tree,t_conts,l_conts,sample = get_contributions(maxslice4, model,df_trn)
        mstmp = sample[topfeatures].describe(include ='all').T
        st.markdown(f"<div><span class='fontgrey'>4. Max category from 4th important contributor: {feats4}</span></div>", unsafe_allow_html=True)
        with st.expander("About this data sample"):
            st.write(mstmp)

        with st.expander("Details of Indv contributions towards the mean prediction target"):
            print_contribs(bias,prediction,feats4)

            fig = go.Figure(go.Waterfall(
                name = "chart", orientation = "h",
                y = df_tree['feature'],
                x = df_tree['cont'],
                connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}}
            ))
            title = f"Feature Contributions related to MAX sample - \"{feats4}\""
            fig.update_layout(title = title, height=500, width=1000)
            st.write(fig)

        bias,prediction,df_tree,t_conts,l_conts,sample = get_contributions(maxslice5, model,df_trn)
        mstmp = sample[topfeatures].describe(include ='all').T
        st.markdown(f"<div><span class='fontgrey'>5. Max category from 5th important contributor: {feats5}</span></div>", unsafe_allow_html=True)
        with st.expander("About this data sample"):
            st.write(mstmp)

        with st.expander("Details of Indv contributions towards the mean prediction target"):
            print_contribs(bias,prediction,feats5)

            fig = go.Figure(go.Waterfall(
                name = "chart", orientation = "h",
                y = df_tree['feature'],
                x = df_tree['cont'],
                connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}}
            ))
            title = f"Feature Contributions related to MAX sample - \"{feats5}\""
            fig.update_layout(title = title, height=500, width=1000)
            st.write(fig)

        #"""**Partial Dependence Plots**"""
        st.markdown("<div><span class='fontgreen'><span class='bold'>2.5 Change Analysis - Simulated Impact of Features on Target</span></div>", unsafe_allow_html=True)
        st.write(f"* Scenario is simulated to help us understand in which direction features are influenzing the target. what happens when the contributor data stays constant and every other column remains as is - will the target value improves/degrades? This data will provide us view on whether the contributing features has really impacted the target value or is it just part of larger multiple contributing features combined.")

        with st.expander("Expand"):
            if rows < 5000:
              x = get_sample(df_trn, 200)
            elif rows < 10000:
              x = get_sample(df_trn,500)
            else:
              x = get_sample(df_trn, 1000)

            if not attr1:
              feat_name = feats1
            else:
              feat_name = feats1+'*'+attr1
            fig = plot_pdp(model, x, feat_name = feat_name, clusters=10)
            st.pyplot(fig)

            if not attr2:
              feat_name = feats2
            else:
              feat_name = feats2+'*'+attr2
            fig=plot_pdp(model, x, feat_name = feat_name, clusters=10)
            st.pyplot(fig)

            if not attr3:
              feat_name = feats3
            else:
              feat_name = feats3+'*'+attr3
            fig=plot_pdp(model, x, feat_name = feat_name, clusters=10)
            st.pyplot(fig)

            if not attr4:
              feat_name = feats4
            else:
              feat_name = feats4+'*'+attr4
            fig=plot_pdp(model, x, feat_name = feat_name, clusters=10)
            st.pyplot(fig)

            if not attr5:
              feat_name = feats5
            else:
              feat_name = feats5+'*'+attr5
            fig=plot_pdp(model, x, feat_name = feat_name, clusters=10)
            st.pyplot(fig)

        st.markdown("<div><span class='fontgreen'><span class='bold'>2.6 Summary of Predicted Insights</span></div>", unsafe_allow_html=True)
        st.write(f"For the chosen target data: {target_variable}, following are the important contributors ")
        st.write("Further summary of insights...WIP")
        st.dataframe(topfeat5n)

else:
    st.error("Upload any dataset in csv format in the sidebar")
    st.markdown("<div><span class='fontgreen'><span class='bold'>About insightsR</span></div>", unsafe_allow_html=True )
    with st.expander("Short Note", expanded=True):
        st.write("**insightsR provides automated insights for tabular data**. Insights will be statistically generated based on the target column (dependent variable) selected by the user. Currently insightsR is limited to selecting dependent variable that has continuous values (Regression) - for eg) total sales, price, cost, income, salary etc.")
        st.write("**User is only expected to provide a dataset in csv format, select a numeric Target column and click 'Generate Insights' button** in the left pane. Rest is taken care by the tool")
    with st.expander("Insights provided and how to interpret", expanded=True):
        st.markdown("Once the user uploads the data to be analysed and provides the target column against which insights to be provided, insightsR tool will handle **pre-processing of data** which includes removal of null columns, highly correlated columns, columns that has more than 75% null values, identifies and  converts date time columns to multiple columns that helps in providing better insights. This pre-processing step is covered in <span class='fontcoral'><b>Section 2.1</b></span>", unsafe_allow_html=True)
        st.write("After pre-processing, insightsR tool will statistically analyse the dataset using machine learning algorithms and provides insights below:-")
        st.markdown(f"* Provides **top contributing features against the target data column**. For eg) if target is 'Sales' features that might turn important is price, seasonality etc.  insightsR tool provides such highly important features and % of contribution. This is covered in <span class='fontcoral'><b>Section 2.2</b></span>", unsafe_allow_html=True)
        st.markdown(f"* Provides **data visualizations for top 5 contributing features**. insightsR tool puts numeric features into multiple buckets to provide a different perception by aligning data into bucketed categories. This is covered in <span class='fontcoral'><b>Section 2.3</b></span>", unsafe_allow_html=True)
        st.markdown(f"* In <span class='fontcoral'><b>Section 2.4</b></span>, deeper insights are provided based on identified top contributors. Based on a sample picked from the top contributors, insightsR first provides **info on whether the contribution is +ve or -ve and by what %**. Features that brought down/up the mean target value is an immensely useful insight. Further within the sample how did each feature contribute is detailed.", unsafe_allow_html=True)
        st.markdown(f"* In <span class='fontcoral'><b>Section 2.5</b></span>, tool **simulates a scenario:** what happens when the contributor data stays constant and every other column data remains as is - will the target value improves/degrades? This data will provide us view on whether the contributing features has really impacted the target value or is it just part of larger multiple contributing features combined.", unsafe_allow_html=True)
        st.write(f"Based on the insights provided, user could identify areas that can be improved, devise a plan on what would happen if a change is simulated, devise a plan to achieve the target.")
