# ── BLOCK 1: IMPORTS ──
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
import os
os.makedirs('outputs', exist_ok=True)
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")



# ── BLOCK 2: LOAD & EXPLORE DATA ──
df = pd.read_csv('cardio_train.csv', sep=';')

print(f"Shape of dataset: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['cardio'].value_counts()}")




# ── BLOCK 3: CLEAN DATA ──

# Drop ID column — it's just a serial number, useless for prediction
df.drop('id', axis=1, inplace=True)

# Convert age from days to years (more meaningful)
df['age'] = (df['age'] / 365).round(1)

# Remove rows with impossible blood pressure values
# A human cannot have negative BP, or systolic lower than diastolic
before = len(df)
df = df[df['ap_hi'] >= df['ap_lo']]
df = df[(df['ap_hi'] > 60) & (df['ap_lo'] > 40)]
df = df[(df['ap_hi'] < 250) & (df['ap_lo'] < 180)]

# Remove impossible height and weight values
df = df[(df['height'] > 100) & (df['height'] < 220)]
df = df[(df['weight'] > 30) & (df['weight'] < 200)]

after = len(df)
print(f"Rows removed as outliers: {before - after}")
print(f"Clean dataset shape: {df.shape}")
print(f"\nAge after conversion (years):\n{df['age'].describe()}")




# ── BLOCK 4: FEATURE ENGINEERING ──

# Create BMI (Body Mass Index) = weight(kg) / height(m)²
df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)

# Create Pulse Pressure = difference between systolic and diastolic BP
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

print(f"New features added. Dataset shape now: {df.shape}")
print(f"\nBMI stats:\n{df['bmi'].describe()}")
print(f"\nPulse Pressure stats:\n{df['pulse_pressure'].describe()}")
print(f"\nFinal columns:\n{df.columns.tolist()}")




# ── BLOCK 5: EDA - VISUALIZATIONS ──
sns.set_theme(style='whitegrid', font_scale=1.1)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Cardiovascular Disease - Exploratory Data Analysis', 
             fontsize=15, fontweight='bold')

# Plot 1 — Target distribution (how many have disease vs not)
counts = df['cardio'].value_counts()
axes[0,0].bar(['No Disease', 'Disease'], counts.values, 
              color=['#55A868', '#DD8452'], edgecolor='white', linewidth=1.5)
for i, v in enumerate(counts.values):
    axes[0,0].text(i, v + 200, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                   ha='center', fontsize=10)
axes[0,0].set_title('Target Distribution')
axes[0,0].set_ylabel('Count')

# Plot 2 — Age distribution by disease outcome
df[df['cardio']==0]['age'].plot(kind='hist', bins=30, alpha=0.6,
    color='#55A868', label='No Disease', ax=axes[0,1])
df[df['cardio']==1]['age'].plot(kind='hist', bins=30, alpha=0.6,
    color='#DD8452', label='Disease', ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Outcome')
axes[0,1].set_xlabel('Age (years)')
axes[0,1].legend()

# Plot 3 — BMI distribution by disease outcome
df[df['cardio']==0]['bmi'].clip(0, 55).plot(kind='hist', bins=30, alpha=0.6,
    color='#55A868', label='No Disease', ax=axes[0,2])
df[df['cardio']==1]['bmi'].clip(0, 55).plot(kind='hist', bins=30, alpha=0.6,
    color='#DD8452', label='Disease', ax=axes[0,2])
axes[0,2].set_title('BMI Distribution by Outcome')
axes[0,2].set_xlabel('BMI')
axes[0,2].legend()

# Plot 4 — Systolic Blood Pressure boxplot
bp = axes[1,0].boxplot(
    [df[df['cardio']==0]['ap_hi'].values, df[df['cardio']==1]['ap_hi'].values],
    labels=['No Disease', 'Disease'], patch_artist=True, notch=True)
bp['boxes'][0].set_facecolor('#55A86888')
bp['boxes'][1].set_facecolor('#DD845288')
axes[1,0].set_title('Systolic Blood Pressure by Outcome')
axes[1,0].set_ylabel('ap_hi (mmHg)')

# Plot 5 — Cholesterol levels vs disease
chol_tab = df.groupby(['cholesterol', 'cardio']).size().unstack()
chol_tab.plot(kind='bar', ax=axes[1,1], color=['#55A868', '#DD8452'],
              edgecolor='white', linewidth=1)
axes[1,1].set_xticklabels(['Normal', 'Above Normal', 'Well Above'], rotation=0)
axes[1,1].set_title('Cholesterol Level vs Outcome')
axes[1,1].set_ylabel('Count')
axes[1,1].legend(['No Disease', 'Disease'])





# Plot 6 — Correlation heatmap
corr = df[['age','bmi','ap_hi','ap_lo','cholesterol','gluc','cardio']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[1,2], linewidths=0.5, annot_kws={'size': 9})
axes[1,2].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig('outputs/eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA plot saved to outputs/eda_analysis.png")




# ── BLOCK 6: SPLIT & SCALE ──

# Separate features (X) from target (y)
X = df.drop('cardio', axis=1)
y = df['cardio']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeatures being used:\n{X.columns.tolist()}")

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% for testing
    stratify=y,          # keep same disease ratio in both splits
    random_state=42      # so results are same every run
)

print(f"\nTraining set size : {X_train.shape[0]:,} patients")
print(f"Testing set size  : {X_test.shape[0]:,} patients")

# Scale the features — CRITICAL for KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # learn mean/std FROM train, then scale
X_test  = scaler.transform(X_test)       # scale test using SAME mean/std from train

print(f"\nScaling done!")
print(f"Mean of first feature before scaling was: {df.drop('cardio',axis=1).iloc[:,0].mean():.2f}")
print(f"Mean of first feature after scaling     : {X_train[:,0].mean():.4f} (close to 0)")

# ── BLOCK 7: FIND BEST K ──
print("\nFinding best K... (this will take a minute, please wait)")

k_range = range(1, 22, 2)  # test K = 1,3,5,7...21 (odd numbers only)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"  K={k:2d}  →  CV Accuracy = {scores.mean():.4f}")

best_k = list(k_range)[np.argmax(cv_scores)]
print(f"\n★ Best K = {best_k}  (accuracy = {max(cv_scores):.4f})")

# Plot elbow curve
plt.figure(figsize=(10, 5))
plt.plot(list(k_range), cv_scores, marker='o', color='#4C72B0', 
         linewidth=2, markersize=8, label='CV Accuracy')
plt.axvline(x=best_k, color='#C44E52', linestyle='--', 
            linewidth=2, label=f'Best K = {best_k}')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Elbow Curve — Finding Optimal K')
plt.legend()
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig('outputs/elbow_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("Elbow curve saved to outputs/elbow_curve.png")

# ── BLOCK 8: TRAIN FINAL MODEL ──
print("\nTraining final KNN model with K =", best_k)

knn_final = KNeighborsClassifier(n_neighbors=best_k, algorithm='ball_tree', n_jobs=-1)
knn_final.fit(X_train, y_train)

y_pred  = knn_final.predict(X_test)
y_proba = knn_final.predict_proba(X_test)[:, 1]

print("Model trained and predictions made!")

# ── BLOCK 9: EVALUATE ──
acc = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"\n{'='*45}")
print(f"  FINAL RESULTS")
print(f"{'='*45}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"{'='*45}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

# Evaluation plots
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(14, 11))
fig.suptitle(f'KNN Model Evaluation (K={best_k})', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# Plot 1 — Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
    xticklabels=['No Disease', 'Disease'],
    yticklabels=['No Disease', 'Disease'],
    linewidths=1, linecolor='white', annot_kws={'size': 13, 'weight': 'bold'})
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')

# Plot 2 — ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(fpr, tpr, color='#4C72B0', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax2.plot([0,1], [0,1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
ax2.fill_between(fpr, tpr, alpha=0.08, color='#4C72B0')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend(loc='lower right')

# Plot 3 — Per class metrics
rep = classification_report(y_test, y_pred, 
      target_names=['No Disease', 'Disease'], output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
x = np.arange(3)
w = 0.3
ax3 = fig.add_subplot(gs[1, 0])
b1 = ax3.bar(x - w/2, [rep['No Disease'][m] for m in metrics], w,
             label='No Disease', color='#55A868', edgecolor='white')
b2 = ax3.bar(x + w/2, [rep['Disease'][m] for m in metrics], w,
             label='Disease', color='#DD8452', edgecolor='white')
for bar in list(b1) + list(b2):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.2f}', ha='center', fontsize=9)
ax3.set_xticks(x)
ax3.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
ax3.set_ylim(0, 1.08)
ax3.set_title('Per-Class Metrics')
ax3.legend()

# Plot 4 — Summary card
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
summary = (
    f"Model Summary\n"
    f"{'─'*32}\n"
    f"Algorithm      :  KNN\n"
    f"Best K         :  {best_k}\n"
    f"Distance Metric:  Euclidean\n"
    f"Train samples  :  {X_train.shape[0]:,}\n"
    f"Test samples   :  {X_test.shape[0]:,}\n"
    f"Features used  :  {X_train.shape[1]}\n"
    f"{'─'*32}\n"
    f"Test Accuracy  :  {acc*100:.2f}%\n"
    f"ROC-AUC        :  {roc_auc:.4f}\n"
    f"Precision(avg) :  {rep['macro avg']['precision']:.4f}\n"
    f"Recall (avg)   :  {rep['macro avg']['recall']:.4f}\n"
    f"F1-Score (avg) :  {rep['macro avg']['f1-score']:.4f}\n"
)
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#F0F4FF',
                   edgecolor='#4C72B0', linewidth=1.5))

plt.savefig('outputs/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Evaluation plot saved to outputs/model_evaluation.png")

# ── BLOCK 10: SAVE MODEL ──
import pickle

# Save the trained model
with open('outputs/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_final, f)

# Save the scaler (MUST save this too — needed for any new predictions)
with open('outputs/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model saved to outputs/knn_model.pkl")
print("Scaler saved to outputs/scaler.pkl")
print("\nTo use the model later on a new patient:")
print("""
    import pickle
    model  = pickle.load(open('outputs/knn_model.pkl', 'rb'))
    scaler = pickle.load(open('outputs/scaler.pkl', 'rb'))

    # New patient data:
    # [age, gender, height, weight, ap_hi, ap_lo,
    #  cholesterol, gluc, smoke, alco, active, bmi, pulse_pressure]
    patient = [[55, 1, 165, 70, 130, 85, 1, 1, 0, 0, 1, 25.7, 45]]
    patient_scaled = scaler.transform(patient)
    prediction = model.predict(patient_scaled)
    probability = model.predict_proba(patient_scaled)[0][1]
    print(f'Disease: {\"Yes\" if prediction[0]==1 else \"No\"}')
    print(f'Probability: {probability:.2%}')
""")

print("\n" + "="*45)
print("  PROJECT COMPLETE!")
print("="*45)
print(f"  outputs/eda_analysis.png      - EDA plots")
print(f"  outputs/elbow_curve.png       - Best K selection")
print(f"  outputs/model_evaluation.png  - Final results")
print(f"  outputs/knn_model.pkl         - Trained model")
print(f"  outputs/scaler.pkl            - Fitted scaler")
print("="*45)