import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
FILE=BASE/"outputs/segmented_customers.csv"
st.set_page_config(page_title="Customer Segmentation",page_icon="👥",layout="wide")
st.title("👥 Customer Segmentation Dashboard")
if not FILE.exists():
    st.error("Run python main.py first."); st.stop()
df=pd.read_csv(FILE)
opts=["All"]+sorted(df.Segment.unique())
sel=st.sidebar.selectbox("Segment",opts)
d=df if sel=="All" else df[df.Segment==sel]
a,b,c,e=st.columns(4)
a.metric("Customers",len(d)); b.metric("Avg Spend",f"₹{d.TotalSpend.mean():,.0f}"); c.metric("Avg Frequency",f"{d.PurchaseFrequency.mean():.1f}"); e.metric("Avg Recency",f"{d.RecencyDays.mean():.1f} days")
st.subheader("Segment Distribution")
fig,ax=plt.subplots(figsize=(9,4)); df.Segment.value_counts().plot(kind="bar",ax=ax); ax.set_xlabel("Segment"); ax.set_ylabel("Customers"); ax.tick_params(axis="x",rotation=20); st.pyplot(fig)
st.subheader("Purchase Frequency vs Total Spend")
fig,ax=plt.subplots(figsize=(9,5))
for s in sorted(df.Segment.unique()):
    q=df[df.Segment==s]; ax.scatter(q.PurchaseFrequency,q.TotalSpend,s=25,alpha=.55,label=s)
ax.set_xlabel("Purchase Frequency"); ax.set_ylabel("Total Spend"); ax.legend(fontsize=8); st.pyplot(fig)
st.subheader("Segment Profile")
st.dataframe(d.groupby("Segment")[["Age","AnnualIncome","PurchaseFrequency","AverageOrderValue","TotalSpend","RecencyDays"]].mean().round(2),use_container_width=True)
st.subheader("Customer Records"); st.dataframe(d,use_container_width=True)
