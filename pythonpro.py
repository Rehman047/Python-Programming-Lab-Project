import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from pandas.tseries.offsets import DateOffset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

# Page Configuration

st.set_page_config(
    page_title="Bank Statement Analyzer",
    page_icon="💳",
    layout="wide"
)


st.markdown("""
<style>
.plot-warning {
    color: #856404;
    background-color: #fff3cd;
    border: 1px solid #ffeeba;
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 1em;
}
.download-btn {
    margin-top: 0.5em;
}
</style>
""", unsafe_allow_html=True)

# ——————————————————————————————
# Default Category Keywords (can be extended by user)
# ——————————————————————————————
CATEGORIES = {
    'Food': ['lulu', 'spinneys', 'zomato', 'market'],
    'Travel': ['uber', 'emirates', 'etihad', 'booking', 'hilton'],
    'Entertainment': ['netflix', 'amazon', 'noon', 'apple'],
    'Bills': ['adcb', 'bank fee', 'insurance'],
    'Other': []
}


# Helper Functions

def prepare(df):
    """Clean amounts and parse dates."""
    df['Amount'] = df['Amount'].astype(str).str.replace(',', '').astype(float)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    return df.dropna(subset=['Date'])

def categorize(df, categories):
    """Assign each transaction a category based on keywords."""
    df = df.copy()
    df['Category'] = 'Other'
    for cat, kws in categories.items():
        if not kws: continue
        mask = df['Details'].str.lower().str.contains('|'.join(kws))
        df.loc[mask, 'Category'] = cat
    return df

def detect_anomaly(df):
    """Flag unusually large debit transactions."""
    df['Anomaly'] = False
    debits = df[df['Debit/Credit']=='Debit']
    if not debits.empty:
        iso = IsolationForest(contamination=0.1, random_state=42)
        preds = iso.fit_predict(debits[['Amount']])
        df.loc[debits.index, 'Anomaly'] = preds == -1
    return df

def plot_and_download(fn, *args, **kw):
    """
    Wraps a plotting function, renders it, and offers a download button.
    """
    try:
        fig, ax = plt.subplots()
        fn(ax, *args, **kw)
        # prevent overlapping date labels
        fig.autofmt_xdate()
        st.pyplot(fig)
        # create PNG buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            "📥 Download chart",
            buf,
            file_name="chart.png",
            mime="image/png",
            key=f"dl_{np.random.randint(0,1e6)}",
            help="Download this chart as a PNG",
        )
        plt.close(fig)
    except Exception as e:
        st.markdown(f"<div class='plot-warning'>Plot error: {e}</div>", unsafe_allow_html=True)


# Sidebar: Data Upload, Year & Date Filters, Custom Categories

st.sidebar.header("1) Upload & Filters")
uploaded = st.sidebar.file_uploader("• Upload your bank CSV", type='csv', help="Exported from your banking app")

if uploaded:
    df_raw = pd.read_csv(uploaded)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], dayfirst=True, errors='coerce')
    min_date, max_date = df_raw['Date'].min(), df_raw['Date'].max()
    date_range = st.sidebar.date_input(
        "• Select date range",
        value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )
else:
    date_range = None

st.sidebar.header("2) Analysis Year(s)")
year1 = st.sidebar.selectbox("Primary year", list(range(2021, 2026)), index=3)
year2 = st.sidebar.selectbox("Compare to year", list([None] + list(range(2021, 2026))), index=0)

st.sidebar.header("3) Add Custom Category")
new_cat = st.sidebar.text_input("Category name")
new_kws = st.sidebar.text_area("Keywords (comma-separated)")
if st.sidebar.button("➕ Add"):
    if new_cat and new_kws:
        CATEGORIES[new_cat] = [kw.strip().lower() for kw in new_kws.split(',')]
        st.sidebar.success(f"Added category '{new_cat}'.")


# Main Title & Description

st.title("🏦 Interactive Bank Statement Analyzer")
st.markdown("""
Welcome! Upload your CSV bank statement, select filters and years, and explore:
- **Spending breakdown** by category and over time  
- **Trends & comparisons** month-to-month or year-over-year  
- **ML insights**: forecast next months, spot anomalies  
- **Custom categories**: add your own on the fly  
- **Download** both data and charts  
""")

if uploaded:
    df = prepare(df_raw)
    df = categorize(df, CATEGORIES)
    df = detect_anomaly(df)

    start_date, end_date = date_range
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    df = df.loc[mask]

    df_y1 = df[df['Date'].dt.year == year1]
    df_y2 = df[df['Date'].dt.year == year2] if year2 else None

    def split_cd(df_):
        return df_[df_['Debit/Credit']=='Credit'], df_[df_['Debit/Credit']=='Debit']

    cr1, db1 = split_cd(df_y1)
    cr2, db2 = split_cd(df_y2) if df_y2 is not None else (None, None)

    # Summary Metrics
    st.header(f"Summary for {year1}" + (f" vs {year2}" if year2 else ""))
    cols = st.columns(4)
    cols[0].metric("Transactions", len(df_y1))
    cols[1].metric("Total Credits", f"AED {cr1['Amount'].sum():,.2f}")
    cols[2].metric("Total Debits", f"AED {db1['Amount'].sum():,.2f}")
    avg_spend = db1.set_index('Date').resample('M')['Amount'].sum().mean()
    cols[3].metric("Avg Monthly Spend", f"AED {avg_spend:,.2f}")

    # Tabs
    tabs = st.tabs(["🧮 Analysis","📈 Trends","🤖 ML Insights","📂 Data Explorer"])

    # Analysis Tab
    with tabs[0]:
        st.subheader("Spending Breakdown by Category & Month")
        if not db1.empty:
            cA, cB = st.columns(2)
            with cA:
                data = db1.groupby('Category')['Amount'].sum()
                plot_and_download(lambda ax: ax.pie(data, labels=data.index, autopct='%1.1f%%'))
            with cB:
                monthly = db1.set_index('Date').resample('M')['Amount'].sum()
                labels = [d.strftime('%b') for d in monthly.index]
                plot_and_download(lambda ax: ax.bar(labels, monthly.values))
        else:
            st.info(f"No debits for {year1} in this date range.")

    # Trends Tab
    with tabs[1]:
        st.subheader("Financial Trends & Year-Over-Year Comparison")
        view = st.radio("Choose view", ["Monthly","Category","Credit vs Debit","Comparison"], horizontal=True)
        if view == "Monthly":
            if not db1.empty:
                series = db1.set_index('Date').resample('M')['Amount'].sum()
                plot_and_download(lambda ax: ax.plot(series.index, series.values, marker='o'))
            else:
                st.info("No data.")
        elif view == "Category":
            cat = st.selectbox("Category", list(CATEGORIES.keys()))
            sub = db1[db1['Category']==cat]
            if not sub.empty:
                grp = sub.set_index('Date').resample('M')['Amount'].sum()
                plot_and_download(lambda ax: ax.plot(grp.index, grp.values, marker='o'))
            else:
                st.info(f"No {cat} spending.")
        elif view == "Credit vs Debit":
            vals = [cr1['Amount'].sum(), db1['Amount'].sum()]
            plot_and_download(lambda ax: ax.pie(vals, labels=["Credits","Debits"], autopct='%1.1f%%'))
        else:  
            if df_y2 is None:
                st.warning("Select a second year to compare.")
            else:
                m1 = db1.set_index('Date').resample('M')['Amount'].sum()
                m2 = db2.set_index('Date').resample('M')['Amount'].sum()
                df_cmp = pd.DataFrame({f"{year1}": m1, f"{year2}": m2}).fillna(0)
                plot_and_download(lambda ax: df_cmp.plot(kind='bar', ax=ax))
                st.caption("Month-to-month comparison between the two selected years")

    # ML Insights Tab
    with tabs[2]:
        st.subheader("Forecast & Anomaly Detection")
        mode = st.selectbox("Mode", ["Expense Forecast","Anomaly Detection"])
        if mode == "Expense Forecast":
            m = db1.set_index('Date').resample('M')['Amount'].sum().reset_index()
            m['Date'] = pd.to_datetime(m['Date'].dt.strftime('%Y-%m-%d'))
            if len(m) >= 3:
                m['Idx'] = np.arange(len(m))
                model = LinearRegression().fit(m[['Idx']], m['Amount'])
                last_date = m['Date'].max()
                future = [last_date + DateOffset(months=i+1) for i in range(3)]
                preds = model.predict(np.arange(len(m), len(m)+3).reshape(-1,1))
                plot_and_download(lambda ax: ax.plot(
                    list(m['Date']) + future,
                    list(m['Amount']) + list(preds),
                    marker='o'
                ))
            else:
                st.warning("Need at least 3 months of data to forecast.")
        else:
            anoms = df_y1[df_y1['Anomaly']]
            if not anoms.empty:
                st.dataframe(anoms[['Date','Details','Amount','Currency']])
            else:
                st.success("No anomalies detected.")

    # Data Explorer Tab
    with tabs[3]:
        st.subheader("View & Download Data")
        view = st.selectbox("Filter rows", ['All','Credits','Debits','Anomalies'])
        if view == 'All':
            st.dataframe(df_y1)
        elif view == 'Credits':
            st.dataframe(cr1)
        elif view == 'Debits':
            st.dataframe(db1)
        else:
            st.dataframe(df_y1[df_y1['Anomaly']])
        st.download_button(
            "📥 Download filtered data CSV",
            df_y1.to_csv(index=False),
            file_name=f"bank_data_{year1}.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a CSV file on the left to begin your analysis.")
