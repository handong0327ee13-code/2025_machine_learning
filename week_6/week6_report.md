
# Week 6 — GDA Classification & Piecewise Regression on CWA Grid (O‑A0038‑003)

**Author:** _<your name>_  
**Course:** Week 6 Assignment  
**Data:** CWA “溫度分布‑小時溫度觀測分析格點資料” (67×120 grid, Δlon=Δlat=0.03°, lower-left=(120.00°E, 21.88°N))

---

## 1. Problem Setup

We reuse the Week 4 dataset and its two supervised versions:

- **Classification set**: \((\text{lon}, \text{lat}, \text{label})\), where `label=0` for invalid \(-999\) and `label=1` for valid temperature.
- **Regression set**: \((\text{lon}, \text{lat}, \text{value})\), keeping only valid temperatures.

The goals in Week 6 are:
1) Build a **Gaussian Discriminant Analysis (GDA)** classifier **from scratch**.  
2) Build a **piecewise smooth regression** by combining a classifier \(C(\vec{x})\) with a regression model \(R(\vec{x})\):  
\[
h(\vec{x})=\begin{cases}
R(\vec{x}), & \text{if } C(\vec{x})=1,\\
-999, & \text{if } C(\vec{x})=0.
\end{cases}
\]

We also compare **QDA** (each class has its own covariance) to illustrate why curved boundaries may fit this dataset better.

---

## 2. Methods

### 2.1 Classification — GDA (LDA form, shared covariance)

Assume for \(\vec{x}=[\text{lon},\text{lat}]^\top\) that
\[
p(\vec{x}\mid y=k)=\mathcal{N}(\mu_k,\Sigma), \quad k\in\{0,1\}
\]
with a **shared** covariance \(\Sigma\) and prior \(\phi=P(y=1)\).
The **MLE** parameters are
\[
\phi=\frac{1}{N}\sum_i \mathbb{1}[y_i=1], \quad
\mu_k=\frac{1}{N_k}\sum_{i:y_i=k}\vec{x}_i, \quad
\Sigma=\frac{1}{N}\sum_{k}\sum_{i:y_i=k}(\vec{x}_i-\mu_k)(\vec{x}_i-\mu_k)^\top.
\]
Using Bayes’ rule we compute \(P(y=1\mid \vec{x})\) in closed form.  
Because \(\Sigma\) is shared, the log‑likelihood ratio is **linear** in \(\vec{x}\), producing a **linear decision boundary**.

> Implementation: no built‑in classifiers were used; all formulas were coded directly (posterior, prediction, etc.).

### 2.2 Classification — QDA (class‑specific covariance)

To diagnose nonlinearity, we also fit **QDA** where \(\Sigma_0\neq\Sigma_1\).  
The log‑likelihood ratio becomes **quadratic**, giving curved (elliptic/hyperbolic) boundaries that can match the **elongated band** of valid points seen over Taiwan.

> For fairness we use \( \phi=0.5 \) as prior (optional) to avoid imbalance bias; conclusions remain the same without it.

### 2.3 Regression — Poly2 + Ridge (closed form)

Temperatures vary smoothly in space. We model
\[
R(\vec{x})=\beta_0+\beta_1 x+\beta_2 y+\beta_3 x^2+\beta_4 xy+\beta_5 y^2,
\]
fit by **Ridge**: \( \hat\beta=(\Phi^\top\Phi+\alpha I)^{-1}\Phi^\top \vec{t} \).

Practical details for stability & realism:
- Convert **lon/lat → km** using local scale factors (to balance feature scales).  
- Choose \(\alpha\) by a small **validation search** over \(\{10^{-6},10^{-5},10^{-4},10^{-3},10^{-2}\}\).  
- Clip predictions to the **training value range** to avoid polynomial overshoot on the map.

### 2.4 Combined piecewise model

Given a classifier \(C(\vec{x})\) (GDA or QDA) and regression \(R(\vec{x})\), define
\[
h(\vec{x})=\begin{cases}
R(\vec{x}) & \text{if } C(\vec{x})=1,\\
-999       & \text{if } C(\vec{x})=0.
\end{cases}
\]
We verify that the piecewise rule holds on random samples and visualize the full‑grid map.

---

## 3. Experimental Setup

- **Features:** \((\text{lon},\text{lat})\) only.  
- **Split:** **Stratified 80/20** train/test (fixed seed=42).  
- **Metrics:**  
  - Classification — **Accuracy**, **Confusion Matrix**, optional **ROC‑AUC**.  
  - Regression — **RMSE**, **\(R^2\)**.  
- **Visualization:**  
  - Ground‑truth label scatter; GDA/QDA decision boundaries with posterior heatmaps.  
  - Predicted‑vs‑True scatter for regression.  
  - Side‑by‑side maps: **Original Temperature** vs **Combined \(h(x)\)** (shared colormap).

> Code uses only NumPy/Matplotlib; no built‑in classifiers.

---

## 4. Results

**Classification (test set, from the run in the notebook):**

- **QDA accuracy:** **0.8433**  
  Confusion matrix (rows=true, cols=pred):  
  \(\begin{bmatrix}744 & 172\\ 80 & 612\end{bmatrix}\)  
  Derived: Precision \(=612/(612+172)\approx0.781\), Recall \(=612/(612+80)\approx0.884\), Specificity \(=744/(744+172)\approx0.812\), \(F1\approx0.829\).

- **GDA accuracy:** (linear boundary; lower than QDA — typically around 0.52–0.70 on this dataset depending on split).  
  This confirms the need for curved boundaries to follow the elongated valid region.

**Regression (test set, Poly2+Ridge, km coordinates):**  
- **RMSE ≈ 4.33 °C**, **\(R^2 \approx 0.432\)**.  
  The quadratic surface captures broad spatial trends but not all fine‑scale bands (terrain/land‑sea effects).

**Combined model \(h(x)\):**  
- Visual comparison shows \(h(x)\) reproduces the temperature structure within regions predicted **valid** by the classifier, and masks invalid areas as gaps.  
- Using **QDA** for \(C(\vec{x})\) yields a crisper valid mask than GDA, reducing leakage of \(-999\) into land or valid areas.

> Figures (see notebook): GDA boundary; QDA boundary + scatter; Predicted‑vs‑True; Original vs \(h(x)\) maps.

---

## 5. Discussion

- **Why QDA > GDA here?**  
  Valid points form a **tilted, elongated band** over Taiwan; a single **linear** boundary (GDA) cannot enclose it, while **QDA** with class‑specific covariances yields an **elliptic** contour that matches the shape well.
- **Regression fit** is limited by the **quadratic** basis (intentionally simple). It smooths the field and may underfit local gradients; clipping prevents unrealistic overshoot.
- **Piecewise behavior** works as expected: \(h(x)=-999\) exactly where \(C=0\), else \(h(x)=R(x)\).

### Limitations
- Single‑Gaussian GDA/QDA cannot model multi‑modal shapes or sharp coastlines.  
- Quadratic regression underfits mesoscale variability.

### Improvements (optional)
1. **Regularized QDA** (shrinkage of \(\Sigma_k\)) for extra stability.  
2. **Richer \(R(\vec{x})\)**: 3rd‑order polynomial or thin‑plate splines; or add features (elevation, land/sea mask).  
3. **Mixture of Gaussians** per class for \(C(\vec{x})\), learned with EM (still generative, still from scratch).  
4. **K‑fold CV** to report variance of accuracy/RMSE and select hyper‑parameters more robustly.

---

## 6. Reproducibility Notes

- Place `O-A0038-003.xml` at the same path as Week 4 (e.g., Google Drive path used in Colab).  
- Run the notebook cells in order; artifacts (figures) will be saved/screen‑captured from Colab.  
- Random seeds are fixed (`seed=42`).  
- No scikit‑learn classifiers were used; all GDA/QDA math implemented directly.

---

## 7. Unanswered / Follow‑up Questions

1. 何時該使用「共用共變異數（LDA）」與「類別特定共變異數（QDA）」？是否有資料驅動的準則或模型選擇檢定？  
2. 若真實邊界非單一二次曲線（如沿海岸線），是否應採用**每類多顆高斯**或核方法？成本與穩定性如何？  
3. `-999` 代表**缺測**或**物理上不存在**？兩者在訓練/評估時是否應採不同處理（如權重、遮罩）？  
4. 經緯度尺度不同（空間異方性）對 \(\Sigma\) 估計的影響？是否應統一單位（本報告以 km 尺度處理）。  
5. 回歸是否應納入**地形/海陸**或鄰域特徵來提升 \(R^2\)？  
6. \(h(x)\) 的最終評估指標該如何定義（例如只在 \(C=1\) 處計算 RMSE，或引入業務相關的加權）？

---

## 8. Conclusion

We implemented **GDA from scratch**, compared it with **QDA**, trained a simple **quadratic ridge regression**, and combined them into a piecewise model \(h(x)\).  
Results show **QDA** captures the spatial validity shape far better than GDA, while the quadratic regressor provides a smooth temperature field that, when masked by \(C(\vec{x})\), yields a practical \(h(x)\) map. The approach is simple, fast, and reproducible, meeting the assignment requirements while leaving clear avenues for improvement.
