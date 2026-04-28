import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from model import preprocessing, clustering

st.title("Clustering App")

file = st.file_uploader("Upload your dataset",type=["csv"])
if file is not None:
    df = pd.read_csv(file)
else:
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

features = st.multiselect(
    "Select features for clustering",
    df.columns.tolist()
)

if len(features)<2:
    st.warning("Select at least 2 features")
    st.stop()

n_clusters = st.slider("Number of clusters",2,10,3)
linkage = st.selectbox("Select linkage",["ward","complete","single"])

X = preprocessing(df,features)
model, labels = clustering(X,n_clusters,linkage)

df["Cluster"] = labels
st.subheader("Clustered Data")
st.dataframe(df)


