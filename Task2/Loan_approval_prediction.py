# ============================================================
# ALFIDO TECH - PROJECT 2: LOAN APPROVAL PREDICTION
# ============================================================

import os,glob,warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score,recall_score,f1_score,
roc_auc_score,confusion_matrix,classification_report,roc_curve,
precision_recall_curve)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ============================================================
# 1. FOLDERS & DATA
# ============================================================

os.makedirs("output",exist_ok=True)
os.makedirs("Visualizations",exist_ok=True)

path="data/loan_dataset.csv"
if not os.path.exists(path):
    files=glob.glob("data/*.csv")
    if not files: raise FileNotFoundError("CSV file not found in data folder.")
    path=files[0]

df=pd.read_csv(path)
df.columns=df.columns.str.strip()

print("="*70)
print("ALFIDO TECH - LOAN APPROVAL PREDICTION")
print("="*70)
print("\nDataset Shape:",df.shape)
print("\nFirst 5 Rows:\n",df.head())
print("\nColumns:\n",df.columns.tolist())

# ============================================================
# 2. CLEANING
# ============================================================

if "Loan_ID" in df.columns: df.drop("Loan_ID",axis=1,inplace=True)

target="Loan_Status"
df[target]=df[target].astype(str).str.strip().str.upper().map({"Y":1,"N":0})
df.dropna(subset=[target],inplace=True)
df.drop_duplicates(inplace=True)
df[target]=df[target].astype(int)

X=df.drop(target,axis=1)
y=df[target]
num=X.select_dtypes(include=np.number).columns.tolist()
cat=X.select_dtypes(exclude=np.number).columns.tolist()

for c in num: X[c]=X[c].fillna(X[c].median())
for c in cat: X[c]=X[c].fillna(X[c].mode()[0])

print("\nMissing Values:\n",X.isnull().sum())

# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(7,5))
sns.countplot(x=y)
plt.title("Loan Approval Class Distribution")
plt.xlabel("Loan Status (0 = Not Approved, 1 = Approved)")
plt.ylabel("Number of Applications")
plt.tight_layout()
plt.savefig("Visualizations/class_distribution.png",dpi=300)
plt.close()

# ============================================================
# 4. TRAIN TEST & PREPROCESSING
# ============================================================

X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=.20,random_state=42,stratify=y)

pre=ColumnTransformer([
("numerical",StandardScaler(),num),
("categorical",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat)])

X_train=pre.fit_transform(X_train)
X_test=pre.transform(X_test)
X_train,y_train=SMOTE(random_state=42).fit_resample(X_train,y_train)

# ============================================================
# 5. MODELS
# ============================================================

models={
"Logistic Regression":LogisticRegression(max_iter=2000,random_state=42),
"Random Forest":RandomForestClassifier(
n_estimators=200,random_state=42,class_weight="balanced")}

results={}
preds={}
probs={}

for name,model in models.items():
    model.fit(X_train,y_train)
    preds[name]=model.predict(X_test)
    probs[name]=model.predict_proba(X_test)[:,1]
    results[name]={
    "Model":name,
    "Precision":precision_score(y_test,preds[name],zero_division=0),
    "Recall":recall_score(y_test,preds[name],zero_division=0),
    "F1 Score":f1_score(y_test,preds[name],zero_division=0),
    "ROC-AUC":roc_auc_score(y_test,probs[name])}

# ============================================================
# 6. MODEL COMPARISON
# ============================================================

result=pd.DataFrame(results.values())
print("\nMODEL COMPARISON")
print(result.to_string(index=False))
result.to_csv("output/model_comparison.csv",index=False)

best=result.loc[result["F1 Score"].idxmax(),"Model"]
pred=preds[best]
prob=probs[best]

print("\nBest Model:",best)

# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

report=classification_report(
y_test,pred,target_names=["Loan Not Approved","Loan Approved"],
zero_division=0)

print("\n",report)

with open("output/classification_report.txt","w") as f:
    f.write(f"Best Model: {best}\n\n{report}")

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm=confusion_matrix(y_test,pred)

plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
xticklabels=["Not Approved","Approved"],
yticklabels=["Not Approved","Approved"])
plt.title(f"{best} - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("Visualizations/confusion_matrix.png",dpi=300)
plt.close()

# ============================================================
# 9. ROC CURVE
# ============================================================

plt.figure(figsize=(8,6))

for name in models:
    fpr,tpr,_=roc_curve(y_test,probs[name])
    auc=roc_auc_score(y_test,probs[name])
    plt.plot(fpr,tpr,label=f"{name} (AUC = {auc:.3f})")

plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("Visualizations/roc_curve.png",dpi=300)
plt.close()

# ============================================================
# 10. PRECISION-RECALL CURVE
# ============================================================

p,r,_=precision_recall_curve(y_test,prob)

plt.figure(figsize=(8,6))
plt.plot(r,p)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve - {best}")
plt.tight_layout()
plt.savefig("Visualizations/precision_recall_curve.png",dpi=300)
plt.close()

# ============================================================
# 11. THRESHOLD ANALYSIS
# ============================================================

thresholds=[.20,.30,.40,.50,.60,.70,.80]
td=[]

for t in thresholds:
    pr=(prob>=t).astype(int)
    td.append({
    "Threshold":t,
    "Precision":precision_score(y_test,pr,zero_division=0),
    "Recall":recall_score(y_test,pr,zero_division=0),
    "F1 Score":f1_score(y_test,pr,zero_division=0)})

td=pd.DataFrame(td)
td.to_csv("output/threshold_analysis.csv",index=False)
best_t=td.loc[td["F1 Score"].idxmax(),"Threshold"]

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

names=pre.get_feature_names_out()

if best=="Random Forest":
    imp=models[best].feature_importances_
else:
    imp=abs(models[best].coef_[0])

fi=pd.DataFrame({"Feature":names,"Importance":imp}).sort_values(
"Importance",ascending=False)

fi.to_csv("output/feature_importance.csv",index=False)

top=fi.head(10).sort_values("Importance")

plt.figure(figsize=(9,6))
plt.barh(top["Feature"],top["Importance"])
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("Visualizations/feature_importance.png",dpi=300)
plt.close()

# ============================================================
# 13. SAVE CLEANED DATA
# ============================================================

df.to_csv("output/cleaned_loan_dataset.csv",index=False)

# ============================================================
# 14. BUSINESS INTERPRETATION
# ============================================================

row=result[result["Model"]==best].iloc[0]

business=f"""
============================================================
LOAN APPROVAL PREDICTION - BUSINESS INTERPRETATION
============================================================

Objective:
Predict whether a loan application is likely to be approved.

Models:
1. Logistic Regression
2. Random Forest

Best Model:
{best}

Recommended Threshold:
{best_t:.2f}

Precision:
{row['Precision']:.4f}

Recall:
{row['Recall']:.4f}

F1 Score:
{row['F1 Score']:.4f}

ROC-AUC:
{row['ROC-AUC']:.4f}

Missing values were handled using median and mode imputation.
Categorical variables were encoded using One-Hot Encoding.
Numerical variables were standardized using StandardScaler.
SMOTE was applied to the training data to handle class imbalance.

A lower threshold can increase recall, while a higher threshold
makes the model more conservative.

The model should support loan screening and should not be the
sole basis for final lending decisions.
"""

with open("output/business_interpretation.txt","w") as f:
    f.write(business)

# ============================================================
# 15. FINAL PROJECT SUMMARY
# ============================================================

summary=f"""
ALFIDO TECH - PROJECT 2
LOAN APPROVAL PREDICTION

Dataset Shape:
{df.shape}

Training Records:
{len(X_train)}

Testing Records:
{len(X_test)}

Best Model:
{best}

Recommended Threshold:
{best_t:.2f}

Precision:
{row['Precision']:.4f}

Recall:
{row['Recall']:.4f}

F1 Score:
{row['F1 Score']:.4f}

ROC-AUC:
{row['ROC-AUC']:.4f}

All required preprocessing, SMOTE, model comparison,
evaluation and threshold analysis were completed.
"""

with open("output/final_project_summary.txt","w") as f:
    f.write(summary)

# ============================================================
# 16. PROJECT COMPLETED
# ============================================================

print("\n"+"="*70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*70)
print("Best Model:",best)
print("Recommended Threshold:",best_t)
print("="*70)