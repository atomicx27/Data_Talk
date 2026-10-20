"""Script to generate a rich, multi-table Excel database for Black-Box Testing."""

from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

def generate_database(output_path: Path) -> dict[str, int]:
    rng = np.random.default_rng(2026)
    
    # 1. Regions (4 rows)
    regions_df = pd.DataFrame({
        "region_id": [1, 2, 3, 4],
        "region_name": ["North America", "EMEA", "Asia Pacific", "Latin America"],
        "country": ["United States", "United Kingdom", "Singapore", "Brazil"],
        "hq_city": ["New York", "London", "Singapore", "Sao Paulo"],
    })
    
    # 2. Sales Reps (10 rows)
    rep_names = [
        "Sarah Jenkins", "Michael Chang", "Elena Rostova", "David Kim",
        "Amara Okafor", "Carlos Gomez", "Priya Sharma", "Lucas Weber",
        "Fatima Al-Mansoor", "Oliver Smith"
    ]
    reps_df = pd.DataFrame({
        "rep_id": list(range(101, 101 + len(rep_names))),
        "rep_name": rep_names,
        "region_id": [1, 1, 2, 2, 3, 3, 4, 1, 2, 3],
        "quota_usd": [1200000, 1000000, 1100000, 950000, 850000, 750000, 600000, 1300000, 900000, 800000],
        "commission_rate": [0.08, 0.07, 0.08, 0.06, 0.07, 0.06, 0.05, 0.09, 0.07, 0.06],
    })
    
    # 3. Products (15 rows)
    products_data = [
        (201, "SKU-CLD-01", "Cloud Data Warehouse", "Cloud Analytics", 400.0, 1200.0),
        (202, "SKU-CLD-02", "Serverless ETL Pipeline", "Cloud Analytics", 250.0, 750.0),
        (203, "SKU-CLD-03", "Real-Time Streaming Engine", "Cloud Analytics", 500.0, 1500.0),
        (204, "SKU-SEC-01", "Zero-Trust Identity Guard", "Cybersecurity", 300.0, 900.0),
        (205, "SKU-SEC-02", "API Endpoint Shield", "Cybersecurity", 180.0, 550.0),
        (206, "SKU-SEC-03", "Data Loss Prevention Suite", "Cybersecurity", 350.0, 1100.0),
        (207, "SKU-DBP-01", "High-Throughput Vector DB", "Database Platform", 600.0, 1800.0),
        (208, "SKU-DBP-02", "Distributed SQL Fabric", "Database Platform", 550.0, 1650.0),
        (209, "SKU-DBP-03", "Embedded Key-Value Cache", "Database Platform", 120.0, 360.0),
        (210, "SKU-AIS-01", "Enterprise LLM Gateway", "AI Solutions", 450.0, 1400.0),
        (211, "SKU-AIS-02", "Autonomous BI Copilot", "AI Solutions", 700.0, 2100.0),
        (212, "SKU-AIS-03", "Vision Quality Inspector", "AI Solutions", 320.0, 980.0),
        (213, "SKU-INT-01", "Kafka Connector Hub", "Integration", 150.0, 450.0),
        (214, "SKU-INT-02", "GraphQL Federation Mesh", "Integration", 200.0, 600.0),
        (215, "SKU-INT-03", "Webhook Event Bus", "Integration", 90.0, 280.0),
    ]
    products_df = pd.DataFrame(products_data, columns=[
        "product_id", "sku_code", "product_name", "category", "unit_cost_usd", "unit_list_price_usd"
    ])
    
    # 4. Customers (60 rows)
    industries = ["Fintech", "Healthcare", "Retail", "Logistics", "Energy", "SaaS"]
    tiers = ["Enterprise", "Mid-Market", "SMB"]
    n_cust = 60
    cust_rows = []
    for i in range(1, n_cust + 1):
        cid = 1000 + i
        tier = rng.choice(tiers, p=[0.25, 0.45, 0.30])
        ind = rng.choice(industries)
        reg_id = int(rng.choice([1, 2, 3, 4], p=[0.40, 0.30, 0.20, 0.10]))
        limit = 500000 if tier == "Enterprise" else (150000 if tier == "Mid-Market" else 40000)
        cust_rows.append({
            "customer_id": cid,
            "company_name": f"Global Client {i:02d} Corp",
            "cust_tier": tier,
            "industry": ind,
            "region_id": reg_id,
            "contact_email": f"exec_client_{i}@enterprise{i}.com",
            "credit_limit": limit,
        })
    customers_df = pd.DataFrame(cust_rows)
    
    # 5. Orders (600 rows)
    n_orders = 600
    order_rows = []
    base_date = date(2024, 1, 15)
    prod_lookup = {row["product_id"]: (row["unit_cost_usd"], row["unit_list_price_usd"]) for _, row in products_df.iterrows()}
    
    for i in range(1, n_orders + 1):
        oid = 5000 + i
        cid = int(rng.choice(customers_df["customer_id"]))
        pid = int(rng.choice(products_df["product_id"]))
        cust_reg = customers_df.loc[customers_df["customer_id"] == cid, "region_id"].values[0]
        # Match rep to customer region if possible
        eligible_reps = reps_df.loc[reps_df["region_id"] == cust_reg, "rep_id"].values
        rep_id = int(rng.choice(eligible_reps) if len(eligible_reps) > 0 else rng.choice(reps_df["rep_id"]))
        
        day_offset = int(rng.integers(0, 750))  # Covers 2024 to mid 2026
        ord_date = (base_date + timedelta(days=day_offset)).isoformat()
        
        stat = rng.choice(["Completed", "Pending", "Cancelled", "On-Hold"], p=[0.74, 0.16, 0.07, 0.03])
        qty = int(rng.integers(1, 15))
        u_cost, u_price = prod_lookup[pid]
        disc = float(rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.45, 0.25, 0.15, 0.10, 0.05]))
        gross = round(qty * u_price, 2)
        net = round(gross * (1.0 - disc), 2)
        cogs = round(qty * u_cost, 2)
        shipping = round(float(rng.uniform(25.0, 180.0)), 2)
        
        order_rows.append({
            "order_id": oid,
            "customer_id": cid,
            "product_id": pid,
            "rep_id": rep_id,
            "order_date": ord_date,
            "ord_stat": stat,
            "qty": qty,
            "gross_revenue_usd": gross,
            "discount_rate": disc,
            "net_revenue_usd": net,
            "cogs_usd": cogs,
            "shipping_usd": shipping,
        })
    orders_df = pd.DataFrame(order_rows)
    
    # 6. Returns (65 rows from Completed orders)
    completed_orders = orders_df[orders_df["ord_stat"] == "Completed"].sample(n=65, random_state=42)
    return_rows = []
    for idx, ord_row in enumerate(completed_orders.itertuples(), start=1):
        ret_id = 9000 + idx
        ord_d = date.fromisoformat(ord_row.order_date)
        ret_d = (ord_d + timedelta(days=int(rng.integers(5, 30)))).isoformat()
        reason = rng.choice(["Defective", "Wrong Item", "Late Delivery", "Customer Request"], p=[0.35, 0.25, 0.20, 0.20])
        refund_pct = float(rng.choice([0.80, 0.90, 1.0]))
        refund_amt = round(ord_row.net_revenue_usd * refund_pct, 2)
        
        return_rows.append({
            "return_id": ret_id,
            "order_id": ord_row.order_id,
            "customer_id": ord_row.customer_id,
            "product_id": ord_row.product_id,
            "return_date": ret_d,
            "ret_reason_cd": reason,
            "refund_amount_usd": refund_amt,
        })
    returns_df = pd.DataFrame(return_rows)
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        regions_df.to_excel(writer, sheet_name="dim_regions", index=False)
        reps_df.to_excel(writer, sheet_name="dim_sales_reps", index=False)
        products_df.to_excel(writer, sheet_name="dim_products", index=False)
        customers_df.to_excel(writer, sheet_name="dim_customers", index=False)
        orders_df.to_excel(writer, sheet_name="fact_orders", index=False)
        returns_df.to_excel(writer, sheet_name="fact_returns", index=False)
        
    counts = {
        "dim_regions": len(regions_df),
        "dim_sales_reps": len(reps_df),
        "dim_products": len(products_df),
        "dim_customers": len(customers_df),
        "fact_orders": len(orders_df),
        "fact_returns": len(returns_df),
    }
    return counts

if __name__ == "__main__":
    out = Path("blackbox_test_run/enterprise_sales_bi.xlsx")
    counts = generate_database(out)
    print(f"Generated Excel database at {out} with {len(counts)} sheets:")
    for k, v in counts.items():
        print(f"  - {k}: {v} rows")
