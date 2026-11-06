"""XGBoost sales forecasting portfolio UI.
Uses explicitly labelled synthetic demo data until the user uploads train.csv.
"""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.metrics import mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor

st.set_page_config(page_title="Sales Forecasting | XGBoost",page_icon="📈",layout="wide",initial_sidebar_state="collapsed")
css=Path(__file__).with_name("sales_theme.css")
if css.exists():st.markdown("<style>"+css.read_text(encoding="utf-8")+"</style>",unsafe_allow_html=True)

def sample_data():
    r=np.random.default_rng(46)
    ds=pd.date_range("2024-01-01",periods=740,freq="D")
    t=np.arange(len(ds))
    y=np.maximum(100,3200+2.1*t+350*np.sin(2*np.pi*t/7)+490*np.sin(2*np.pi*t/365)+r.normal(0,265,len(t)))
    return pd.DataFrame({"Date":ds,"Sales":np.round(y,2)})

def uploaded_data(blob,ext):
    f=io.BytesIO(blob)
    raw=pd.read_excel(f) if ext in ("xlsx","xls") else pd.read_csv(f)
    normalized={str(c).strip().lower():c for c in raw.columns}
    date_col=next((normalized[k] for k in ("order date","date","order_date","ds","transaction_date") if k in normalized),None)
    sales_col=next((normalized[k] for k in ("sales","revenue","y","amount","total_sales") if k in normalized),None)
    if not date_col or not sales_col:raise ValueError("The file needs Order Date (or Date) and Sales columns.")
    d=raw[[date_col,sales_col]].rename(columns={date_col:"Date",sales_col:"Sales"}).copy()
    d["Date"]=pd.to_datetime(d["Date"],errors="coerce",dayfirst=True).dt.normalize()
    d["Sales"]=pd.to_numeric(d["Sales"],errors="coerce")
    d=d.dropna().groupby("Date",as_index=False)["Sales"].sum().sort_values("Date")
    d=d.set_index("Date").resample("D").sum().reset_index()
    if len(d)<65 or len(d)>7000:raise ValueError("Upload between 65 and 7,000 days of sales history.")
    return d

LAGS=(1,2,3,4,5,7,14)
FEATURES=[f"lag_{n}" for n in LAGS]+["rolling_7","rolling_14","weekday","month","dayofyear","trend"]
def create_features(d):
    x=d.copy()
    for lag in LAGS:x[f"lag_{lag}"]=x["Sales"].shift(lag)
    x["rolling_7"]=x["Sales"].shift(1).rolling(7).mean()
    x["rolling_14"]=x["Sales"].shift(1).rolling(14).mean()
    x["weekday"]=x["Date"].dt.weekday
    x["month"]=x["Date"].dt.month
    x["dayofyear"]=x["Date"].dt.dayofyear
    x["trend"]=np.arange(len(x))
    return x.dropna().reset_index(drop=True)

@st.cache_data(show_spinner=False)
def prepare(blob,ext,horizon,n_trees):
    raw=sample_data() if ext=="demo" else uploaded_data(blob,ext)
    f=create_features(raw)
    split=int(len(f)*.8)
    a,b=f.iloc[:split],f.iloc[split:]
    if len(a)<35 or len(b)<8:raise ValueError("Not enough observations for a holdout evaluation.")
    model=XGBRegressor(n_estimators=n_trees,max_depth=4,learning_rate=.065,subsample=.9,colsample_bytree=.9,random_state=42,n_jobs=1,objective="reg:squarederror")
    model.fit(a[FEATURES],a["Sales"])
    predicted=np.clip(model.predict(b[FEATURES]),0,None)
    rmse=float(np.sqrt(mean_squared_error(b["Sales"],predicted)))
    mae=float(mean_absolute_error(b["Sales"],predicted))
    holdout=pd.DataFrame({"Date":b["Date"],"Actual":b["Sales"],"Predicted":predicted})
    model.fit(f[FEATURES],f["Sales"])
    history=raw["Sales"].astype(float).tolist()
    start=raw["Date"].iloc[-1]
    future=[]
    for t in range(1,horizon+1):
        date=start+pd.Timedelta(days=t)
        row={f"lag_{n}":history[-n] for n in LAGS}
        row.update({"rolling_7":float(np.mean(history[-7:])),"rolling_14":float(np.mean(history[-14:])),"weekday":date.weekday(),"month":date.month,"dayofyear":date.dayofyear,"trend":len(history)})
        val=max(0.,float(model.predict(pd.DataFrame([row],columns=FEATURES))[0]))
        history.append(val)
        future.append({"Date":date,"Forecast":val})
    importance=pd.DataFrame({"Feature":FEATURES,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False).head(10)
    return raw,pd.DataFrame(future),holdout,importance,rmse,mae

st.markdown("""<div class="hero"><div><span class="eyebrow">RETAIL ANALYTICS / PREDICTIVE MACHINE LEARNING</span><h1>Sales Forecasting <em>Dashboard</em></h1><p>Train on sales history, explore demand patterns and forecast future revenue with XGBoost.</p><div class="hero-tags"><span>XGBoost</span><span>Time-series forecasting</span><span>Interactive modeling</span></div></div><div class="hero-symbol">↗</div></div>""",unsafe_allow_html=True)
with st.container(border=True):
    c1,c2,c3=st.columns([1.6,1,1])
    with c1:source=st.radio("Data source",["Synthetic sample","Upload sales data"],horizontal=True)
    with c2:horizon=st.select_slider("Forecast horizon (days)",options=[7,14,21,28,45,60,90],value=28)
    with c3:n_trees=st.select_slider("XGBoost trees",options=[60,100,140,200],value=140)
ext="demo";blob=b""
if source=="Upload sales data":
    uploaded=st.file_uploader("Upload CSV/Excel · columns Order Date and Sales",type=["csv","xlsx","xls"])
    if not uploaded:
        st.info("Upload a dataset to enable real-data training and forecasting.")
        st.stop()
    ext=uploaded.name.rsplit(".",1)[-1].lower()
    blob=uploaded.getvalue()
else:
    st.info("**SYNTHETIC SAMPLE DATA** — the notebook references train.csv, which is not included in the repository. These are illustrative artificial sales, not actual company records.")

try:
    with st.spinner("Training and forecasting…"):
        history,future,holdout,importance,rmse,mae=prepare(blob,ext,horizon,n_trees)
except Exception as e:
    st.error(f"Forecast unavailable: {e}")
    st.stop()

tabs=st.tabs(["◈  Forecast Overview","▤  Model Evaluation","▥  Data & Drivers","ⓘ  Methodology"])
with tabs[0]:
    a,b,c,d=st.columns(4)
    a.metric("Sales · last 30 days",f"{history['Sales'].tail(30).sum():,.0f}")
    b.metric("Forecast revenue",f"{future['Forecast'].sum():,.0f}")
    c.metric("Days of history",f"{len(history):,}")
    d.metric("Holdout RMSE",f"{rmse:,.0f}")
    left,right=st.columns([1.6,1],gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("#### Sales history & future forecast")
            h=history.tail(110).rename(columns={"Sales":"Value"}).assign(Series="Historical")
            f=future.rename(columns={"Forecast":"Value"}).assign(Series="Forecast")
            dta=pd.concat([h,f],ignore_index=True)
            chart=alt.Chart(dta).mark_line(strokeWidth=3,interpolate="monotone",point=False).encode(
                x=alt.X("Date:T",title=None,axis=alt.Axis(format="%b %d",labelAngle=0)),
                y=alt.Y("Value:Q",title="Sales"),color=alt.Color("Series:N",scale=alt.Scale(domain=["Historical","Forecast"],range=["#546bd0","#e4a458"])),
                tooltip=[alt.Tooltip("Date:T",format="%b %d, %Y"),alt.Tooltip("Value:Q",format=",.0f"),"Series:N"]
            ).properties(height=280).configure_view(stroke=None).configure_axis(gridColor="#e8edf4",domain=False,labelColor="#61738a",titleColor="#61738a",tickColor="#e8edf4").configure_legend(labelColor="#33475e",titleColor="#33475e",orient="top")
            st.altair_chart(chart,use_container_width=True,theme=None)
    with right:
        with st.container(border=True):
            st.markdown("#### Forecast by week")
            byweek=future.assign(Week=lambda x:(np.arange(len(x))//7)+1).groupby("Week",as_index=False)["Forecast"].sum()
            fig=alt.Chart(byweek).mark_bar(color="#486fa4",cornerRadiusTopLeft=8,cornerRadiusTopRight=8,size=38).encode(
                x=alt.X("Week:O",title="Week"),y=alt.Y("Forecast:Q",title="Forecast total"),
                tooltip=["Week:O",alt.Tooltip("Forecast:Q",format=",.0f")]
            ).properties(height=280).configure_view(stroke=None).configure_axis(gridColor="#e8edf4",domain=False,labelColor="#61738a",titleColor="#61738a",tickColor=False).configure_axisX(labelAngle=0)
            st.altair_chart(fig,use_container_width=True,theme=None)
    st.download_button("↓ Export forecast CSV",future.to_csv(index=False).encode(),"xgboost_forecast.csv","text/csv",type="primary")
with tabs[1]:
    a,b=st.columns(2)
    a.metric("Holdout RMSE",f"{rmse:,.2f}")
    b.metric("Holdout MAE",f"{mae:,.2f}")
    with st.container(border=True):
        st.markdown("#### Actual vs predicted · final 20%")
        plot=holdout.melt("Date",value_vars=["Actual","Predicted"],var_name="Series",value_name="Sales")
        st.altair_chart(alt.Chart(plot).mark_line(strokeWidth=2).encode(
            x=alt.X("Date:T",title=None),y=alt.Y("Sales:Q"),
            color=alt.Color("Series:N",scale=alt.Scale(domain=["Actual","Predicted"],range=["#546bd0","#1ba89c"])),
            tooltip=["Date:T",alt.Tooltip("Sales:Q",format=",.0f"),"Series:N"]
        ).properties(height=300).configure_view(stroke=None).configure_axis(gridColor="#e8edf4",domain=False,labelColor="#61738a",titleColor="#61738a").configure_legend(orient="top"),use_container_width=True,theme=None)
    st.caption("Time-ordered holdout uses observed previous sales as lagged features. Future multi-step predictions reuse model outputs and may be less accurate.")
with tabs[2]:
    left,right=st.columns(2,gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("#### Feature importance")
            st.altair_chart(alt.Chart(importance.sort_values("Importance")).mark_bar(color="#6475d0",cornerRadiusEnd=4).encode(
                x=alt.X("Importance:Q"),y=alt.Y("Feature:N",sort=None,title=None),
                tooltip=["Feature:N",alt.Tooltip("Importance:Q",format=".3f")]
            ).properties(height=300).configure_view(stroke=None).configure_axis(gridColor="#e8edf4",domain=False,labelColor="#61738a",titleColor="#61738a"),use_container_width=True,theme=None)
    with right:
        with st.container(border=True):
            st.markdown("#### Latest observations")
            st.dataframe(history.tail(15),use_container_width=True,hide_index=True,height=325)
with tabs[3]:
    st.markdown("#### Methodology")
    st.write("This web app implements XGBoost on date-aggregated daily sales with lag features, 7/14-day rolling means and calendar features, inspired by the original project notebook. Model metrics are computed on the last 20% of the series before the model is refit on all data for forward forecasting.")
    st.warning("The sample dataset is synthetic. Its error scores must not be represented as performance on the original project's missing train.csv.")
    st.link_button("GitHub project","https://github.com/ayoubennadifb-spec/Sales_prediction_XGBOOST")
st.markdown('<div class="footer">SALES FORECASTING / XGBOOST <span>Educational portfolio app · data source identified in the interface</span></div>',unsafe_allow_html=True)
