import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from model import preprocessing, clustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


fig, ax = plt.subplots()
pca = PCA(n_components=2)
numerical = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numerical])
 
X_pca = pca.fit_transform(X_scaled)


ax.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["Cluster"]
)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
st.pyplot(fig)

st.subheader("Clustered Data")
st.dataframe(df)


