# Final Clean Database Schema (Post-ETL + Predictions)

This document describes the **final clean tables** that live in the **analytics Postgres** database (and are mirrored to Snowflake). These are the tables your Agentic RAG will query using **SELECT-only SQL**.

## Overview
- Source raw dumps (Supabase): orders line items + subscription events.
- ETL (Spark) normalizes into dimensions + facts.
- MLOps inference writes outputs to separate prediction tables.
- All timestamps are stored as **UTC** (or timezone-aware `timestamptz`).

### Entity Relationship Summary
- `customers (1) -> (M) orders`
- `orders (1) -> (M) order_items`
- `products (1) -> (M) order_items`
- `customers (1) -> (M) subscriptions`
- `customers (1) -> (M) churn_predictions`
- `products (1) -> (M) forecast_predictions`

---

## Table: customers
**Purpose:** Customer dimension. Email is unique but not a primary key.

**Columns**
- `customer_id` (text) **PK**
- `email` (text) **UNIQUE**
- `full_name` (text)
- `address` (text)
- `city` (text)
- `state` (char(2))
- `zip_code` (text)
- `region` (text) — one of: West, Midwest, South, East

**Indexes**
- `PRIMARY KEY (customer_id)`
- `UNIQUE (email)`
- `INDEX (region)`
- `INDEX (state)`

---

## Table: products
**Purpose:** Product/SKU dimension. Raw data contains `sku`; ETL generates `product_id` as a surrogate key.

**Columns**
- `product_id` (text) **PK**
- `sku` (text) **UNIQUE**
- `product_name` (text)
- `category` (text)
- `base_price_usd` (numeric(10,2))

**Indexes**
- `PRIMARY KEY (product_id)`
- `UNIQUE (sku)`
- `INDEX (category)`

---

## Table: orders
**Purpose:** Order header facts.

**Columns**
- `order_id` (text) **PK**
- `customer_id` (text) **FK -> customers(customer_id)**
- `order_timestamp` (timestamptz)
- `promo_code` (text, nullable)

**Indexes**
- `PRIMARY KEY (order_id)`
- `INDEX (customer_id)`
- `INDEX (order_timestamp)`
- Recommended composite: `INDEX (customer_id, order_timestamp DESC)` for “latest orders per customer”.

---

## Table: order_items
**Purpose:** Order line items (one row per SKU item within an order).

**Columns**
- `order_item_id` (text) **PK**
- `order_id` (text) **FK -> orders(order_id)**
- `product_id` (text) **FK -> products(product_id)**
- `quantity` (int)
- `unit_price_usd` (numeric(10,2))

**Indexes**
- `PRIMARY KEY (order_item_id)`
- `INDEX (order_id)` for order -> items lookups
- `INDEX (product_id)` for product-level rollups
- Recommended composite: `INDEX (product_id, order_id)` for “product sales by order”.

---

## Table: subscriptions
**Purpose:** Clean subscription events/snapshots used for churn features and analytics.

**Columns**
- `subscription_id` (text) **PK**
- `customer_id` (text) **FK -> customers(customer_id)**
- `event_timestamp` (timestamptz)
- `plan_name` (text)
- `billing_cycle` (text)
- `plan_price_usd` (numeric(10,2))
- `status` (text)
- `last_billed_timestamp` (timestamptz)
- `failed_payments_30d` (int)
- `autopay_enabled` (boolean)
- `support_tickets_30d` (int)
- `state` (char(2))

**Indexes**
- `PRIMARY KEY (subscription_id)`
- `INDEX (customer_id)`
- `INDEX (event_timestamp)`
- Recommended composite: `INDEX (customer_id, event_timestamp DESC)` for “latest subscription state”.
- Optional partial index: `INDEX (status) WHERE status IN ('past_due','canceled')`

---

## Table: churn_predictions
**Purpose:** Batch churn inference outputs (separate from subscriptions).

**Columns**
- `snapshot_month` (date)
- `customer_id` (text) **FK -> customers(customer_id)**
- `churn_probability` (numeric(6,4))
- `churn_flag` (boolean)
- `model_name` (text)

**Keys / Indexes**
- Recommended **UPSERT key**: `UNIQUE (customer_id, snapshot_month, model_name)`
- `INDEX (snapshot_month)`
- `INDEX (churn_flag)`
- Recommended composite: `INDEX (snapshot_month, churn_flag)` for “high risk counts per month”.

---

## Table: forecast_predictions
**Purpose:** Next-month daily demand forecasts (separate from orders).

**Columns**
- `forecast_date` (date)
- `region` (text)
- `product_id` (text) **FK -> products(product_id)**
- `predicted_units` (numeric(10,2))
- `predicted_units_lower` (numeric(10,2), nullable)
- `predicted_units_upper` (numeric(10,2), nullable)
- `model_name` (text)

**Keys / Indexes**
- Recommended **UPSERT key**: `UNIQUE (product_id, region, forecast_date, model_name)`
- `INDEX (forecast_date)`
- `INDEX (region)`
- Recommended composite: `INDEX (forecast_date, region, product_id)` for dashboards.

---

## Performance Recommendations for RAG Querying
1) **Foreign key indexes:** Ensure every FK column has an index (`orders.customer_id`, `order_items.order_id`, `order_items.product_id`, `subscriptions.customer_id`, etc.).
2) **Composite indexes for common question patterns:**
   - Customer-centric: `(customer_id, order_timestamp DESC)`
   - Product-centric: `(product_id, order_id)` and forecast `(product_id, region, forecast_date)`
3) **Partitioning (optional, later):**
   - Range partition `orders` by month on `order_timestamp`
   - Range partition `forecast_predictions` by month on `forecast_date`
4) **Materialized views (optional):**
   - daily_sales_by_product_region
   - churn_rate_by_plan_month
5) **RAG guardrails:** expose read-only schema to the agent; enforce `SELECT` only, block `INSERT/UPDATE/DELETE/DDL`.

---

## Example JOIN Queries (for agent testing)
### Forecast by category and region (next month)
```sql
SELECT p.category, fp.region, fp.forecast_date, SUM(fp.predicted_units) AS units
FROM forecast_predictions fp
JOIN products p ON p.product_id = fp.product_id
WHERE fp.forecast_date BETWEEN '2025-12-01' AND '2025-12-31'
GROUP BY p.category, fp.region, fp.forecast_date
ORDER BY fp.forecast_date, p.category;
```

### High churn customers by plan
```sql
SELECT s.plan_name, COUNT(*) AS high_risk_customers
FROM churn_predictions cp
JOIN subscriptions s ON s.customer_id = cp.customer_id
WHERE cp.snapshot_month = '2025-12-01' AND cp.churn_flag = TRUE
GROUP BY s.plan_name
ORDER BY high_risk_customers DESC;
```
