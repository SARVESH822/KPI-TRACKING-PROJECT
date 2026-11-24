"""
Generates a realistic synthetic structured dataset (daily order-level sales)
and a matching unstructured dataset (support tickets / sales notes / ops notes)
with intentionally injected business events so the analytics engine has
something real to discover.

Hero scenario (do not change without updating docs/demo-script.md):
  - South India / Electronics revenue drops in August 2026 due to:
      * order volume decline (fulfillment bottleneck)
      * rising delivery delays
      * reduced promotional activity
  - West India revenue rises in August 2026 due to a promotion push
  - Margin falls in West India despite the revenue rise, due to discounting
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
STRUCT_DIR = ROOT / "data" / "structured"
UNSTRUCT_DIR = ROOT / "data" / "unstructured"
STRUCT_DIR.mkdir(parents=True, exist_ok=True)
UNSTRUCT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ["South India", "North India", "West India", "East India"]
CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Grocery"]
CHANNELS = ["App", "Web", "Marketplace"]
SEGMENTS = ["New", "Returning", "Loyalty"]

BASE_PRICE = {"Electronics": 4200, "Apparel": 1200, "Home & Kitchen": 1800, "Beauty": 650, "Grocery": 400}
BASE_DAILY_ORDERS = {  # per region per category, roughly
    "South India": 1.15, "North India": 1.30, "West India": 1.20, "East India": 0.85,
}

START = date(2026, 6, 1)
END = date(2026, 8, 31)

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

rows = []
for d in daterange(START, END):
    is_august = d.month == 8
    for region in REGIONS:
        for category in CATEGORIES:
            for channel in CHANNELS:
                for segment in SEGMENTS:
                    lam = 3.2 * BASE_DAILY_ORDERS[region]
                    # category/channel/segment weighting so not all cells are equal
                    lam *= {"Electronics": 1.3, "Apparel": 1.1, "Home & Kitchen": 1.0,
                            "Beauty": 0.8, "Grocery": 1.4}[category]
                    lam *= {"App": 1.2, "Web": 1.0, "Marketplace": 0.7}[channel]
                    lam *= {"New": 0.9, "Returning": 1.1, "Loyalty": 0.85}[segment]

                    discount = np.random.uniform(0.03, 0.12)
                    delivery_days = np.random.normal(3.2, 0.5)
                    conversion = np.random.uniform(0.018, 0.032)
                    returns_rate = np.random.uniform(0.02, 0.06)

                    # --- Event 1: South India Electronics decline in August ---
                    if is_august and region == "South India" and category == "Electronics":
                        lam *= 0.72          # order volume down ~28%
                        delivery_days += np.random.uniform(1.5, 2.8)  # delivery delays up
                        discount *= 0.55     # promo activity down
                        conversion *= 0.85   # conversion softens

                    # --- Event 2: West India promo-driven uplift in August ---
                    if is_august and region == "West India":
                        lam *= 1.35
                        discount = min(0.28, discount + np.random.uniform(0.10, 0.16))
                        conversion *= 1.15

                    orders = np.random.poisson(max(lam, 0.05))
                    if orders == 0:
                        continue

                    price = BASE_PRICE[category] * np.random.uniform(0.92, 1.08)
                    units_per_order = np.random.choice([1, 1, 1, 2, 2, 3])
                    units = orders * units_per_order
                    gross_revenue = units * price
                    discount_amount = gross_revenue * discount
                    net_revenue = gross_revenue - discount_amount
                    cogs_ratio = np.random.uniform(0.55, 0.68)
                    margin = net_revenue - (gross_revenue * cogs_ratio)

                    rows.append({
                        "date": d.isoformat(),
                        "region": region,
                        "category": category,
                        "channel": channel,
                        "segment": segment,
                        "orders": int(orders),
                        "units": int(units),
                        "price": round(price, 2),
                        "discount_rate": round(discount, 4),
                        "revenue": round(net_revenue, 2),
                        "margin": round(margin, 2),
                        "conversion_rate": round(conversion, 4),
                        "delivery_days": round(max(delivery_days, 0.5), 2),
                        "returns_rate": round(returns_rate, 4),
                    })

df = pd.DataFrame(rows)
df.to_csv(STRUCT_DIR / "sales.csv", index=False)
print(f"structured rows: {len(df)}")

# ---------------- Unstructured documents ----------------
docs = []
doc_id = 1

def add_doc(d, region, category, source, title, text):
    global doc_id
    docs.append({
        "document_id": f"DOC-{doc_id:04d}",
        "date": d.isoformat(),
        "region": region,
        "category": category,
        "source": source,
        "title": title,
        "text": text,
    })
    doc_id += 1

# July baseline complaints (low volume) for South India Electronics
july_days = [d for d in daterange(date(2026, 7, 1), date(2026, 7, 31))]
for d in random.sample(july_days, 13):
    add_doc(d, "South India", "Electronics", "Support Ticket",
            "Delivery delay",
            "Customer reports order arrived later than the estimated delivery date. "
            "Requesting update on dispatch status.")

# August: 31 delivery-delay tickets (2.4x) concentrated in South India Electronics
august_days = [d for d in daterange(date(2026, 8, 1), date(2026, 8, 31))]
for d in random.sample(august_days * 2, 31):
    add_doc(d, "South India", "Electronics", "Support Ticket",
            "Delayed shipment complaint",
            "Customer escalated a delayed shipment. Package has been in transit "
            "beyond the promised delivery window from the regional distribution center.")

# Ops incident notes pointing at a specific distribution center
for d in random.sample(august_days, 6):
    add_doc(d, "South India", "Electronics", "Operations Note",
            "DC-South capacity constraint",
            "Distribution Center South (DC-South) is running below normal throughput "
            "due to a staffing shortfall on the outbound sorting line, adding roughly "
            "2 days to average dispatch time for Electronics SKUs.")

# Sales rep notes: reduced Electronics promotions in South
for d in random.sample(august_days, 4):
    add_doc(d, "South India", "Electronics", "Sales Note",
            "Promotion budget reallocated",
            "Regional promotion budget for Electronics was reduced this month and "
            "reallocated to the West region campaign; South Electronics has no active "
            "discount push in market.")

# Management commentary on West India promo push
for d in random.sample(august_days, 5):
    add_doc(d, "West India", None, "Management Note",
            "West region festival promotion",
            "Launched a festival promotion campaign across West India covering multiple "
            "categories with increased discount depth to drive volume ahead of the season.")

# Finance/ops note on margin erosion in West
for d in random.sample(august_days, 3):
    add_doc(d, "West India", None, "Finance Note",
            "Margin compression flagged",
            "Finance flags that the West region promotion is driving revenue growth but "
            "gross margin percentage has declined due to deeper discounting than planned.")

# Some noise documents unrelated to the hero scenario, other regions/categories
noise_topics = [
    ("North India", "Apparel", "Support Ticket", "Size exchange request",
     "Customer requesting a size exchange for an apparel order, unrelated to delivery timing."),
    ("East India", "Grocery", "Support Ticket", "Missing item in order",
     "Customer reports one item missing from a grocery order, requesting refund."),
    ("North India", "Beauty", "Sales Note", "Regular restock",
     "Routine restock note for beauty category, no anomalies observed."),
]
for region, category, source, title, text in noise_topics:
    for d in random.sample(august_days, 4):
        add_doc(d, region, category, source, title, text)

docs_df = pd.DataFrame(docs)
docs_df.to_csv(UNSTRUCT_DIR / "documents.csv", index=False)
with open(UNSTRUCT_DIR / "documents.json", "w") as f:
    json.dump(docs, f, indent=2)
print(f"unstructured docs: {len(docs_df)}")
