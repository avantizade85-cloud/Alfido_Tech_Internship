# ============================================================
# ALFIDO TECH INTERNSHIP
# PROJECT: ZOMATO RESTAURANT & REVIEW DATA ANALYSIS
# ============================================================

import os,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
warnings.filterwarnings("ignore")

OUT,VIZ="output","visualizations"
os.makedirs(OUT,exist_ok=True);os.makedirs(VIZ,exist_ok=True)

# 1. PROJECT DIRECTORIES
# 2. FIND DATASET
files=[f for f in os.listdir("data") if f.endswith(".csv")]
if not files:
    print("ERROR: No CSV file found!");exit()
DATA="data/"+files[0]

# 3. LOAD DATA
try: df=pd.read_csv(DATA,encoding="utf-8")
except: df=pd.read_csv(DATA,encoding="latin1")

print("="*70)
print("ALFIDO TECH - ZOMATO DATA ANALYSIS")
print("="*70)
print("\nDataset:",DATA)
print("\nDataset loaded successfully.")
print("\nOriginal Shape:\n",df.shape)
print("\nColumns:")
for c in df.columns: print("-",c)

# 4. STANDARDIZE COLUMN NAMES
df.columns=df.columns.str.strip().str.replace("\n"," ",regex=False)

# 5. BASIC DATA INFORMATION
print("\n"+"="*70)
print("DATA INFORMATION")
print("="*70)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nDataset Information:")
df.info()

# 6. MISSING VALUES
print("\n"+"="*70)
print("MISSING VALUES")
print("="*70)
m=df.isnull().sum()
mt=pd.DataFrame({
    "Column":m.index,
    "Missing Values":m.values,
    "Missing Percentage":(m.values/len(df)*100).round(2)
})
mt=mt[mt["Missing Values"]>0].sort_values(
    "Missing Values",ascending=False)
print(mt)
mt.to_csv(OUT+"/missing_values_report.csv",index=False)

# 7. REMOVE DUPLICATES
print("\nDuplicate rows:",df.duplicated().sum())
df=df.drop_duplicates()
print("Shape after removing duplicates:",df.shape)

# 8. HANDLE TEXT COLUMNS
for c in df.select_dtypes("object"):
    df[c]=df[c].fillna("Unknown")

# 9. HANDLE NUMERIC COLUMNS
for c in df.select_dtypes(np.number):
    df[c]=df[c].fillna(df[c].median())

# 10. IDENTIFY IMPORTANT COLUMNS
rating=cuisine=city=cost=price=votes=restaurant=delivery=booking=None

for c in df.columns:
    x=c.lower()
    if "rate" in x: rating=c
    if "cuisine" in x: cuisine=c
    if "location" in x or x=="city": city=c
    if "approx_cost" in x: cost=c
    if "price range" in x: price=c
    if "vote" in x: votes=c
    if x=="name": restaurant=c
    if "online_order" in x: delivery=c
    if "book_table" in x: booking=c

print("\n"+"="*70)
print("IMPORTANT COLUMNS DETECTED")
print("="*70)
print("Rating:",rating)
print("Cuisine:",cuisine)
print("City:",city)
print("Cost:",cost)
print("Price Range:",price)
print("Votes:",votes)
print("Restaurant:",restaurant)
print("Online Delivery:",delivery)
print("Table Booking:",booking)

# 11. CONVERT NUMERIC COLUMNS
if rating:
    df[rating]=pd.to_numeric(
        df[rating].astype(str).str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce")

if votes:
    df[votes]=pd.to_numeric(df[votes],errors="coerce")

if cost:
    df[cost]=pd.to_numeric(
        df[cost].astype(str).str.replace(",","",regex=False),
        errors="coerce")

# 12. RATING ANALYSIS
if rating:
    print("\n"+"="*70)
    print("RATING ANALYSIS")
    print("="*70)
    print(df[rating].describe())

    plt.figure(figsize=(10,6))
    sns.histplot(df[rating],bins=20,kde=True)
    plt.title("Distribution of Restaurant Ratings")
    plt.xlabel("Restaurant Rating")
    plt.ylabel("Number of Restaurants")
    plt.tight_layout()
    plt.savefig(VIZ+"/01_rating_distribution.png",dpi=300)
    plt.close()

# 13. CUISINE VS RATING
if cuisine and rating:
    print("\n"+"="*70)
    print("CUISINE VS RATING")
    print("="*70)

    x=df[[cuisine,rating]].copy()
    x[cuisine]=x[cuisine].astype(str).str.split(",")
    x=x.explode(cuisine)
    x[cuisine]=x[cuisine].str.strip()

    cr=x.groupby(cuisine)[rating].agg(
        ["mean","count"]).reset_index()
    cr=cr[cr["count"]>=5].sort_values("mean",ascending=False)

    print("\nTop cuisines by average rating:")
    print(cr.head(10))
    cr.to_csv(OUT+"/cuisine_rating_analysis.csv",index=False)

    plt.figure(figsize=(12,7))
    sns.barplot(data=cr.head(10),x="mean",y=cuisine)
    plt.title("Top 10 Cuisines by Average Rating")
    plt.xlabel("Average Rating")
    plt.ylabel("Cuisine")
    plt.tight_layout()
    plt.savefig(VIZ+"/02_cuisine_vs_rating.png",dpi=300)
    plt.close()

# 14. LOCATION HOTSPOTS
if city:
    print("\n"+"="*70)
    print("LOCATION HOTSPOTS")
    print("="*70)

    cc=df[city].value_counts().head(15)
    print("\nTop restaurant locations:")
    print(cc)
    cc.to_csv(OUT+"/location_hotspots.csv")

    plt.figure(figsize=(12,7))
    sns.barplot(x=cc.values,y=cc.index)
    plt.title("Top Restaurant Locations")
    plt.xlabel("Number of Restaurants")
    plt.ylabel("City")
    plt.tight_layout()
    plt.savefig(VIZ+"/03_location_hotspots.png",dpi=300)
    plt.close()

# 15. LOCATION VS RATING
if city and rating:
    cr=df.groupby(city)[rating].agg(
        ["mean","count"]).reset_index()
    cr=cr[cr["count"]>=5].sort_values("mean",ascending=False)
    cr.to_csv(OUT+"/city_rating_analysis.csv",index=False)

# 16. PRICE VS RATING
if cost and rating:
    pr=df.groupby(pd.qcut(df[cost],4,duplicates="drop"))[rating].agg(
        ["mean","count"]).reset_index()
    pr.to_csv(OUT+"/price_vs_rating_analysis.csv",index=False)

    plt.figure(figsize=(10,6))
    sns.barplot(data=pr,x=pr.columns[0],y="mean")
    plt.title("Price Range vs Average Rating")
    plt.xlabel("Price Range")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(VIZ+"/04_price_vs_rating.png",dpi=300)
    plt.close()

