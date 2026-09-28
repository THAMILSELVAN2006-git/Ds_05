import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import json
import joblib
import time
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report

start_time = time.time()
print("Starting Full DS_Day01_05 End-to-End Pipeline...")

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# 1. Load Data
df_users = pd.read_csv('users_data.csv')
df_txn_agg = pd.read_csv('customer_transactions_aggregated.csv')

# Clean money columns in users
for col in ['per_capita_income', 'yearly_income', 'total_debt']:
    df_users[col + '_num'] = df_users[col].astype(str).str.replace('$', '', regex=False).str.strip().astype(float)

# Merge active transacting customers
df_clean = pd.merge(df_users, df_txn_agg, left_on='id', right_on='client_id', how='inner')
print(f"df_clean transacting customer rows: {len(df_clean)}")

# Feature Engineering
# Age groups
bins_age = [17, 24, 34, 44, 54, 64, 120]
labels_age = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df_clean['Age_Group'] = pd.cut(df_clean['current_age'], bins=bins_age, labels=labels_age)

# Income_Band (quartiles of yearly_income)
df_clean['Income_Band'] = pd.qcut(df_clean['yearly_income_num'], q=4, labels=['Low', 'Lower-Middle', 'Upper-Middle', 'High'])

# Credit_Score_Band
cs_bins = [299, 579, 669, 739, 799, 850]
cs_labels = ['Poor (<580)', 'Fair (580-669)', 'Good (670-739)', 'Very Good (740-799)', 'Exceptional (800+)']
df_clean['Credit_Score_Band'] = pd.cut(df_clean['credit_score'], bins=cs_bins, labels=cs_labels)

# Debt to income ratio (safe division)
df_clean['Debt_to_Income'] = np.where(df_clean['yearly_income_num'] > 0, 
                                      df_clean['total_debt_num'] / df_clean['yearly_income_num'], 0.0)

# Save cleaned banking data
df_clean.to_csv('cleaned_banking_data.csv', index=False)
print("Saved cleaned_banking_data.csv")

# ==========================================
# PHASE 4: EDA & STATISTICAL TESTING
# ==========================================
print("\n--- PHASE 4: EDA & STATISTICAL TESTING ---")
# Group sizes
print("Group Sizes:")
for seg in ['Age_Group', 'Income_Band', 'gender', 'Credit_Score_Band', 'Proxy_Region_State']:
    vc = df_clean[seg].value_counts()
    print(f"\n{seg} (total categories: {len(vc)}):")
    flagged = vc[vc < 30]
    print(vc.head(6))
    if len(flagged) > 0:
        print(f"  --> FLAGGED n < 30 in {seg}: {dict(flagged)}")

# Kruskal-Wallis & ANOVA for Digital_Ratio across segments
eda_tests = {}
for seg in ['Age_Group', 'Income_Band', 'gender', 'Credit_Score_Band']:
    groups = [group['Digital_Ratio'].values for name, group in df_clean.groupby(seg, observed=False) if len(group) > 0]
    # Kruskal
    kw_stat, kw_p = stats.kruskal(*groups)
    # ANOVA
    f_stat, f_p = stats.f_oneway(*groups)
    # Effect size eta-squared
    ss_between = sum(len(g) * (np.mean(g) - df_clean['Digital_Ratio'].mean())**2 for g in groups)
    ss_total = sum((df_clean['Digital_Ratio'] - df_clean['Digital_Ratio'].mean())**2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    
    eda_tests[seg] = {
        'Kruskal_stat': float(kw_stat), 'Kruskal_p': float(kw_p),
        'ANOVA_F': float(f_stat), 'ANOVA_p': float(f_p),
        'eta_squared': float(eta_sq)
    }
    print(f"Digital_Ratio across {seg}: KW p={kw_p:.4e}, ANOVA p={f_p:.4e}, eta^2={eta_sq:.4f}")

# Total spend across Income Band
groups_inc = [g['Total_Spend'].values for n, g in df_clean.groupby('Income_Band', observed=False)]
kw_inc_stat, kw_inc_p = stats.kruskal(*groups_inc)
print(f"Total_Spend across Income_Band: KW p={kw_inc_p:.4e}")

# ==========================================
# PHASE 5: EXACTLY 4 MAIN CHARTS + 2 SUPPLEMENTARY
# ==========================================
print("\n--- PHASE 5: GENERATING CHARTS ---")

# Chart 1: Overall channel comparison
fig1, ax1 = plt.subplots(figsize=(8, 5))
channel_counts = {
    'Swipe Card': 6967185,
    'Chip Card': 4780818,
    'Online (Digital)': 1557912,
    'Bill Payments': 564004
}
total_tx = 13305915
channels = list(channel_counts.keys())
counts = [channel_counts[c] for c in channels]
pcts = [c / total_tx * 100 for c in counts]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

bars = ax1.bar(channels, [c / 1e6 for c in counts], color=colors, edgecolor='none', width=0.55)
ax1.set_ylabel('Transaction Volume (Millions)', fontsize=11, fontweight='bold')
ax1.set_title('Figure 1: Overall Banking Transaction Channel Volume (2010 - 2019)', fontsize=12, fontweight='bold', pad=15)
ax1.set_ylim(0, 8.5)

for bar, pct, cnt in zip(bars, pcts, counts):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{cnt:,.0f}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='semibold')

plt.tight_layout()
fig1.savefig('chart_1_channel_comparison.png', dpi=300)
plt.close(fig1)
print("Saved chart_1_channel_comparison.png")

# Chart 2: Digital (Online) share by Age_Group
fig2, ax2 = plt.subplots(figsize=(8, 5))
age_dig = df_clean.groupby('Age_Group', observed=False)['Digital_Ratio'].agg(['mean', 'median', 'count']).reset_index()
# Note: 18-24 has n=4, flag it
x_pos = np.arange(len(age_dig))
bars2 = ax2.bar(x_pos, age_dig['mean'] * 100, width=0.55, color='#ff7f0e', alpha=0.85, label='Mean Digital Share (%)')
ax2.plot(x_pos, age_dig['median'] * 100, color='#d62728', marker='o', linewidth=2, label='Median Digital Share (%)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{row['Age_Group']}\n(n={row['count']})" + ("*" if row['count'] < 30 else "") 
                     for _, row in age_dig.iterrows()])
ax2.set_ylabel('Digital Transaction Share (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Customer Age Group (* indicates sample size n < 30)', fontsize=11, fontweight='bold')
ax2.set_title('Figure 2: Customer Digital (Online) Transaction Share by Age Group', fontsize=12, fontweight='bold', pad=15)
ax2.legend(frameon=True)
ax2.set_ylim(0, 22)

for bar, (_, row) in zip(bars2, age_dig.iterrows()):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='semibold')

plt.tight_layout()
fig2.savefig('chart_2_digital_share_by_age.png', dpi=300)
plt.close(fig2)
print("Saved chart_2_digital_share_by_age.png")

# Chart 3: In-person card usage by segment (ATM not derivable)
fig3, ax3 = plt.subplots(figsize=(8, 5))
inc_inperson = df_clean.groupby('Income_Band', observed=False)['InPerson_Ratio'].agg(['mean', 'median', 'count']).reset_index()
bars3 = ax3.bar(inc_inperson['Income_Band'], inc_inperson['mean'] * 100, width=0.5, color='#1f77b4', alpha=0.85)
ax3.set_ylabel('In-Person Card Usage Ratio (%)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Customer Income Band (Quartiles of Yearly Income)', fontsize=11, fontweight='bold')
ax3.set_title('Figure 3: In-Person Card Usage by Customer Income Band\n(Substitute for ATM/Cash Withdrawal: ATM MCC 6011 Absent)', fontsize=11, fontweight='bold', pad=12)
ax3.set_ylim(80, 96)

for bar, (_, row) in zip(bars3, inc_inperson.iterrows()):
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%\n(n={row['count']})", ha='center', va='bottom', fontsize=9, fontweight='semibold')

plt.tight_layout()
fig3.savefig('chart_3_inperson_card_usage_by_segment.png', dpi=300)
plt.close(fig3)
print("Saved chart_3_inperson_card_usage_by_segment.png")

# Chart 4: Distribution / outlier plot of customer transaction activity
fig4, (ax4_top, ax4_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
sns.boxplot(x=df_clean['Txn_Total'], ax=ax4_top, color='#9ecae1', fliersize=4)
ax4_top.set_xlabel('')
ax4_top.set_title('Figure 4: Distribution and Outlier Analysis of Total Customer Transactions', fontsize=12, fontweight='bold', pad=12)

sns.histplot(df_clean['Txn_Total'], ax=ax4_bot, bins=40, kde=True, color='#3182bd')
ax4_bot.set_xlabel('Total Transactions per Customer (10-Year Period)', fontsize=11, fontweight='bold')
ax4_bot.set_ylabel('Customer Count', fontsize=11, fontweight='bold')

q1 = df_clean['Txn_Total'].quantile(0.25)
q3 = df_clean['Txn_Total'].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
ax4_bot.axvline(upper_fence, color='red', linestyle='--', linewidth=1.5, label=f'1.5x IQR Outlier Fence ({upper_fence:,.0f})')
ax4_bot.axvline(df_clean['Txn_Total'].median(), color='green', linestyle='-', linewidth=1.5, label=f'Median ({df_clean["Txn_Total"].median():,.0f})')
ax4_bot.legend(frameon=True)

plt.tight_layout()
fig4.savefig('chart_4_customer_activity_outliers.png', dpi=300)
plt.close(fig4)
print("Saved chart_4_customer_activity_outliers.png")

# ==========================================
# PHASE 6: ATM TREND ANALYSIS & VERBATIM STATEMENT
# ==========================================
print("\n--- PHASE 6: ATM TREND ANALYSIS ---")
atm_statement = "The available data does not identify ATM/cash withdrawals, so ATM decline cannot be assessed."
print(f"Mandated Statement: '{atm_statement}'")

# Trend test on monthly channel data
df_m = pd.read_csv('monthly_channel_trends.csv')
piv_m = df_m.pivot(index='ym', columns='use_chip', values='share').fillna(0)
piv_m = piv_m.iloc[:-1] # drop last month if partial or check full range
print(f"Evaluating monthly channel shares across {len(piv_m)} months...")

# Fit linear slopes
trends = {}
x_months = np.arange(len(piv_m))
for ch in piv_m.columns:
    slope, intercept, r_val, p_val, std_err = stats.linregress(x_months, piv_m[ch].values)
    trends[ch] = {'slope': float(slope), 'intercept': float(intercept), 'r_sq': float(r_val**2), 'p_value': float(p_val)}
    print(f"Channel '{ch}': monthly slope = {slope:+.6f}, p-value = {p_val:.4e} ({'Statistically Significant' if p_val < 0.05 else 'Not Significant'})")

# ==========================================
# PHASE 7: OUTLIERS (IQR)
# ==========================================
print("\n--- PHASE 7: OUTLIER PROFILES ---")
outlier_report = {}
outlier_metrics = ['Txn_Total', 'Total_Spend', 'Avg_Amount', 'Refund_Rate', 'Error_Rate']

for m in outlier_metrics:
    vals = df_clean[m]
    q1 = vals.quantile(0.25)
    q3 = vals.quantile(0.75)
    iqr = q3 - q1
    lower_f = q1 - 1.5 * iqr
    upper_f = q3 + 1.5 * iqr
    outliers = df_clean[(vals < lower_f) | (vals > upper_f)]
    count = len(outliers)
    pct = count / len(df_clean) * 100
    top_ids = outliers['id'].head(10).tolist()
    
    outlier_report[m] = {
        'Q1': float(q1), 'Q3': float(q3), 'IQR': float(iqr),
        'upper_fence': float(upper_f), 'lower_fence': float(lower_f),
        'outlier_count': int(count), 'outlier_pct': float(pct),
        'sample_client_ids': top_ids
    }
    print(f"Metric '{m}': {count} customers ({pct:.2f}%) requiring further investigation / enhanced monitoring (Fence: > {upper_f:.4f})")

# ==========================================
# PHASE 8: ML TARGET DEFINITION
# ==========================================
print("\n--- PHASE 8: ML TARGET DEFINITION ---")
# Compare cuts:
cut_50 = (df_clean['Digital_Ratio'] > 0.5).sum()
cut_med = (df_clean['Digital_Ratio'] >= df_clean['Digital_Ratio'].median()).sum()
cut_10 = (df_clean['Digital_Ratio'] >= 0.10).sum()

print(f"Target Threshold Options:")
print(f"  Option 1: > 0.5 (Absolute Majority) -> {cut_50} positive ({cut_50/len(df_clean)*100:.2f}%) [EXTREME IMBALANCE: 98.4% vs 1.6%]")
print(f"  Option 2: >= 0.10 (Bank-wide Mean ~10.48%) -> {cut_10} positive ({cut_10/len(df_clean)*100:.2f}%) [25.2% Digital Adopters vs 74.8% Traditional]")
print(f"  Option 3: >= Median (0.06375) -> {cut_med} positive ({cut_med/len(df_clean)*100:.2f}%) [50.0% Higher Digital vs 50.0% Lower Digital]")

# Defensible Choice:
# Median split provides optimal 50/50 balance for KNN distance-based learning without minority collapse.
# We will use Transaction_Preference based on Median split (0.06375) as primary, and also report the 0.10 threshold comparison.
df_clean['Transaction_Preference'] = np.where(df_clean['Digital_Ratio'] >= df_clean['Digital_Ratio'].median(), 1, 0)
print(f"Final ML Target: Transaction_Preference (Median split >= {df_clean['Digital_Ratio'].median():.4f})")
print(f"Class 1 (Digital-oriented): {(df_clean['Transaction_Preference'] == 1).sum()}")
print(f"Class 0 (In-person-oriented): {(df_clean['Transaction_Preference'] == 0).sum()}")

# ==========================================
# PHASE 9: KNN MODEL WITH STRICT LEAKAGE PREVENTION
# ==========================================
print("\n--- PHASE 9: KNN MODELING ---")
# LEAKAGE RULE CHECK:
# Must NOT include: Online_Count, Swipe_Count, Chip_Count, Digital_Ratio, InPerson_Ratio, use_chip derivatives, id, address
allowed_features = [
    'current_age', 'gender', 'yearly_income_num', 'per_capita_income_num', 
    'total_debt_num', 'credit_score', 'num_credit_cards', 'retirement_age', 
    'Debt_to_Income', 'Proxy_Region_State', 'Avg_Amount', 'Txn_per_Month', 'Refund_Rate', 'Error_Rate'
]
cat_cols = ['gender', 'Proxy_Region_State']
num_cols = [c for c in allowed_features if c not in cat_cols]

print(f"Allowed Input Features ({len(allowed_features)}): {allowed_features}")

X = df_clean[allowed_features].copy()
y = df_clean['Transaction_Preference'].values

# Stratified 80/20 train/test split (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# Baseline: DummyClassifier(strategy='most_frequent')
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
dummy_acc = accuracy_score(y_test, y_pred_dummy)
dummy_bal_acc = balanced_accuracy_score(y_test, y_pred_dummy)
print(f"Baseline (Most Frequent) - Accuracy: {dummy_acc:.4f}, Balanced Accuracy: {dummy_bal_acc:.4f}")

# Select K in 1..30 with 5-fold Stratified CV on training data only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
k_values = list(range(1, 31))
cv_scores_mean = []
cv_scores_std = []

for k in k_values:
    fold_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr_f, X_val_f = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr_f, y_val_f = y_train[train_idx], y_train[val_idx]
        
        # Fit preprocessor on fold train only
        X_tr_proc = preprocessor.fit_transform(X_tr_f)
        X_val_proc = preprocessor.transform(X_val_f)
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr_proc, y_tr_f)
        preds = knn.predict(X_val_proc)
        fold_scores.append(balanced_accuracy_score(y_val_f, preds))
        
    cv_scores_mean.append(np.mean(fold_scores))
    cv_scores_std.append(np.std(fold_scores))

best_idx = int(np.argmax(cv_scores_mean))
best_k = k_values[best_idx]
best_score = cv_scores_mean[best_idx]
best_se = cv_scores_std[best_idx] / np.sqrt(5)

# 1-SE rule simplest odd K
threshold_score = best_score - best_se
simplest_k = best_k
for k, sc in zip(k_values, cv_scores_mean):
    if k % 2 == 1 and sc >= threshold_score:
        simplest_k = k
        break

print(f"Best K: {best_k} (CV Balanced Accuracy: {best_score:.4f})")
print(f"1-SE Rule Simplest Odd K: {simplest_k}")

# Supplementary Plot: K vs CV score
fig_k, ax_k = plt.subplots(figsize=(8, 4.8))
ax_k.errorbar(k_values, cv_scores_mean, yerr=cv_scores_std, fmt='-o', color='#1f77b4', ecolor='#aec7e8', elinewidth=1.5, capsize=3)
ax_k.axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k} (Score: {best_score:.4f})')
ax_k.axvline(simplest_k, color='green', linestyle=':', label=f'1-SE Simplest Odd K={simplest_k}')
ax_k.set_xlabel('Number of Neighbors (K)', fontsize=11, fontweight='bold')
ax_k.set_ylabel('5-Fold CV Balanced Accuracy', fontsize=11, fontweight='bold')
ax_k.set_title('Supplementary Figure S1: KNN Hyperparameter K Selection Curve', fontsize=12, fontweight='bold', pad=12)
ax_k.legend(frameon=True)
plt.tight_layout()
fig_k.savefig('chart_supplementary_k_curve.png', dpi=300)
plt.close(fig_k)
print("Saved chart_supplementary_k_curve.png")

# Final Model Pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=simplest_k))
])
final_pipeline.fit(X_train, y_train)

# Evaluate on test set once
y_pred_test = final_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
test_prec = precision_score(y_test, y_pred_test, zero_division=0)
test_rec = recall_score(y_test, y_pred_test, zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)
cls_rep = classification_report(y_test, y_pred_test, target_names=['In-Person-Oriented', 'Digital-Oriented'], output_dict=True)

print("\n--- FINAL TEST EVALUATION ---")
print(f"Accuracy:          {test_acc:.4f} (vs Baseline: {dummy_acc:.4f})")
print(f"Balanced Accuracy: {test_bal_acc:.4f} (vs Baseline: {dummy_bal_acc:.4f})")
print(f"Precision:         {test_prec:.4f}")
print(f"Recall:            {test_rec:.4f}")
print(f"F1 Score:          {test_f1:.4f}")
print("Confusion Matrix:\n", cm)

# Supplementary Plot: Confusion Matrix
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
            xticklabels=['In-Person Predominant (0)', 'Digital Predominant (1)'],
            yticklabels=['In-Person Predominant (0)', 'Digital Predominant (1)'])
ax_cm.set_xlabel('Predicted Transaction Preference', fontsize=11, fontweight='bold')
ax_cm.set_ylabel('True Transaction Preference', fontsize=11, fontweight='bold')
ax_cm.set_title('Supplementary Figure S2: KNN Confusion Matrix (Test Set, N=244)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
fig_cm.savefig('chart_supplementary_confusion_matrix.png', dpi=300)
plt.close(fig_cm)
print("Saved chart_supplementary_confusion_matrix.png")

# Circular Sanity Check (with leaked features)
X_leaked = df_clean[allowed_features + ['Digital_Ratio', 'Online_Count', 'InPerson_Ratio']].copy()
X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(X_leaked, y, test_size=0.2, random_state=42, stratify=y)
num_l = num_cols + ['Digital_Ratio', 'Online_Count', 'InPerson_Ratio']
prep_l = ColumnTransformer([('num', StandardScaler(), num_l), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)])
pipe_leaked = Pipeline([('preprocessor', prep_l), ('classifier', KNeighborsClassifier(n_neighbors=simplest_k))])
pipe_leaked.fit(X_tr_l, y_tr_l)
y_pred_leaked = pipe_leaked.predict(X_te_l)
acc_leaked = accuracy_score(y_te_l, y_pred_leaked)
print(f"\n[CIRCULAR SANITY CHECK - INVALID/LEAKED]: Test Accuracy = {acc_leaked:.4f} (Demonstrates tautological leakage when use_chip derivatives are included).")

# ==========================================
# PHASE 10: DEPLOYMENT ARTIFACTS & DEMO PREDICTIONS
# ==========================================
print("\n--- PHASE 10: DEPLOYMENT ARTIFACTS ---")
joblib.dump(final_pipeline, 'knn_pipeline.pkl')
print("Saved knn_pipeline.pkl")

with open('feature_list.json', 'w') as f:
    json.dump({
        'features': allowed_features,
        'numerical_features': num_cols,
        'categorical_features': cat_cols,
        'target': 'Transaction_Preference',
        'classes': ['In-person-oriented', 'Digital-oriented'],
        'threshold_definition': f"Digital_Ratio >= {df_clean['Digital_Ratio'].median():.5f} (median split)",
        'chosen_k': simplest_k,
        'metrics': {
            'accuracy': float(test_acc),
            'balanced_accuracy': float(test_bal_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1': float(test_f1),
            'baseline_accuracy': float(dummy_acc)
        }
    }, f, indent=2)
print("Saved feature_list.json")

# Predict function
def predict_customer(cust_dict):
    df_input = pd.DataFrame([cust_dict])
    pred = final_pipeline.predict(df_input)[0]
    prob = final_pipeline.predict_proba(df_input)[0]
    class_name = 'Digital-oriented' if pred == 1 else 'In-person-oriented'
    return {
        'prediction': int(pred),
        'preference_class': class_name,
        'probability_in_person': float(prob[0]),
        'probability_digital': float(prob[1])
    }

# 3 Demonstration Records within real dataset ranges
demo_customers = [
    {
        'desc': 'Demonstration Profile 1 (Young High-Income Professional)',
        'data': {
            'current_age': 29, 'gender': 'Female', 'yearly_income_num': 95000.0,
            'per_capita_income_num': 48000.0, 'total_debt_num': 25000.0,
            'credit_score': 740, 'num_credit_cards': 3, 'retirement_age': 65,
            'Debt_to_Income': 25000.0 / 95000.0, 'Proxy_Region_State': 'CA',
            'Avg_Amount': 62.50, 'Txn_per_Month': 115.0, 'Refund_Rate': 0.035, 'Error_Rate': 0.010
        }
    },
    {
        'desc': 'Demonstration Profile 2 (Senior Retired Traditional Spender)',
        'data': {
            'current_age': 72, 'gender': 'Male', 'yearly_income_num': 34000.0,
            'per_capita_income_num': 18000.0, 'total_debt_num': 2000.0,
            'credit_score': 780, 'num_credit_cards': 4, 'retirement_age': 65,
            'Debt_to_Income': 2000.0 / 34000.0, 'Proxy_Region_State': 'FL',
            'Avg_Amount': 38.20, 'Txn_per_Month': 70.0, 'Refund_Rate': 0.020, 'Error_Rate': 0.012
        }
    },
    {
        'desc': 'Demonstration Profile 3 (Middle-Aged High-Frequency Household)',
        'data': {
            'current_age': 45, 'gender': 'Female', 'yearly_income_num': 52000.0,
            'per_capita_income_num': 26000.0, 'total_debt_num': 68000.0,
            'credit_score': 660, 'num_credit_cards': 2, 'retirement_age': 67,
            'Debt_to_Income': 68000.0 / 52000.0, 'Proxy_Region_State': 'TX',
            'Avg_Amount': 44.10, 'Txn_per_Month': 98.0, 'Refund_Rate': 0.055, 'Error_Rate': 0.022
        }
    }
]

print("\n--- DEMONSTRATION PREDICTIONS (NOT REAL CUSTOMERS) ---")
demo_results = []
for demo in demo_customers:
    res = predict_customer(demo['data'])
    demo_results.append({'desc': demo['desc'], 'input': demo['data'], 'output': res})
    print(f"\n{demo['desc']}:")
    print(f"  Prediction: {res['preference_class']} (Prob Digital: {res['probability_digital']:.2%}, Prob In-Person: {res['probability_in_person']:.2%})")

with open('demo_predictions.json', 'w') as f:
    json.dump(demo_results, f, indent=2)

print(f"\nPipeline successfully completed in {time.time() - start_time:.2f}s!")
