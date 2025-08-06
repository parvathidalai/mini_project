# 🛒 Market Basket Analysis on Instamart Dataset

This project performs Market Basket Analysis on the Instamart dataset using the **Apriori algorithm** to find frequent itemsets and generate association rules.

---

## 📌 Objective

To identify frequently purchased item combinations and generate actionable insights using association rule mining (e.g., "People who buy A often buy B").

---

## 📁 Dataset

- **Source**: Instamart orders (CSV file)
- **Columns Used**:
  - `BillNo` – Unique transaction ID
  - `Itemname` – Product name

---

## 🧠 Techniques Used

- **Data Cleaning**
- **Transaction Grouping**
- **One-Hot Encoding**
- **Apriori Algorithm** (via `mlxtend`)
- **Association Rule Mining**

---

## 🛠️ Libraries Required

Install dependencies using:

```bash
pip install -r requirements.txt
