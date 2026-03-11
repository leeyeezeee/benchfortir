import sys
import os

sys.path.append(os.getcwd())

import json
import random
from typing import Any, Dict, List, Optional

from .base_tool import BaseTool


# ---------------------------------------------------------------------------
# Shared synthetic store data (mirrors interaction/interaction_process.py)
# ---------------------------------------------------------------------------

POLICIES: Dict[str, str] = {
    "returns": (
        "Return policy: Most items can be returned within 30 days of delivery in original condition. "
        "Some categories (e.g., hygiene items) may be non-returnable if opened. "
        "Refunds are processed after the returned item is received and inspected, typically 3-7 business days."
    ),
    "shipping": (
        "Shipping policy: Standard shipping ETA depends on destination and carrier. "
        "Expedited shipping availability varies by region. Address changes may not be possible after shipment."
    ),
    "warranty": (
        "Warranty policy: Manufacturer warranty applies to defects in materials/workmanship. "
        "Accidental damage and normal wear may not be covered. Proof of purchase may be required."
    ),
    "price_match": (
        "Price match policy: Eligible for identical items sold by authorized retailers. "
        "Marketplace listings, auctions, and limited-time flash deals may be excluded."
    ),
    "coupons": (
        "Coupon policy: Coupons may have eligibility constraints and usually cannot be stacked unless explicitly stated."
    ),
}

CATALOG: List[Dict[str, Any]] = [
    {"sku": "ELEC-1001", "name": "Noise-Cancelling Bluetooth Headphones", "price": 129.99, "domain": "electronics"},
    {"sku": "ELEC-1002", "name": "USB-C Fast Charger 65W", "price": 24.99, "domain": "electronics"},
    {"sku": "ELEC-1003", "name": "Portable SSD 1TB", "price": 79.99, "domain": "electronics"},
    {"sku": "HOME-2001", "name": "Air Fryer 5QT", "price": 89.99, "domain": "home_kitchen"},
    {"sku": "HOME-2002", "name": "Robot Vacuum", "price": 199.99, "domain": "home_kitchen"},
    {"sku": "BEAU-3001", "name": "Sunscreen SPF 50", "price": 14.99, "domain": "beauty_personal_care"},
    {"sku": "FASH-4001", "name": "Running Sneakers", "price": 59.99, "domain": "fashion"},
    {"sku": "SPRT-5001", "name": "Adjustable Dumbbells", "price": 149.99, "domain": "sports_outdoors"},
    {"sku": "PET-6001", "name": "Automatic Pet Feeder", "price": 49.99, "domain": "pet_supplies"},
    {"sku": "OFFC-7001", "name": "Ergonomic Office Chair", "price": 229.99, "domain": "office_supplies"},
]

COUPONS: Dict[str, Dict[str, Any]] = {
    "SAVE10": {"type": "percent", "value": 10, "min_subtotal": 50.0},
    "SAVE20": {"type": "percent", "value": 20, "min_subtotal": 150.0},
    "SHIPFREE": {"type": "shipping", "value": 0, "min_subtotal": 35.0},
    "WELCOME5": {"type": "fixed", "value": 5.0, "min_subtotal": 20.0},
}


def _simple_match_score(q: str, name: str) -> float:
    """Token-overlap + substring score, same as interaction_process."""
    q = q.lower().strip()
    name = name.lower().strip()
    if not q:
        return 0.0
    qtoks = set(q.split())
    ntoks = set(name.split())
    overlap = len(qtoks & ntoks) / max(1, len(qtoks))
    substr = 1.0 if q in name else 0.0
    return 0.7 * overlap + 0.3 * substr


def _parse_json_or_wrap(content: str) -> Dict[str, Any]:
    """
    Helper: parse JSON from tag content if possible, otherwise wrap as {"query": content}.

    This allows two calling styles for tag-based tools:
    - <product_search>{"query": "...", "domain": "...", "k": 5}</product_search>
    - <product_search>headphones in electronics</product_search>
    """
    raw = (content or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    # Fallback: treat as simple query text
    return {"query": raw}


# ---------------------------------------------------------------------------
# Tool implementations (tag-based, for ToolExecutor)
# ---------------------------------------------------------------------------


class ProductSearchTool(BaseTool):
    """Tag-based product search: <product_search>...</product_search>."""

    @property
    def name(self) -> str:
        return "product_search"

    @property
    def trigger_tag(self) -> str:
        # The tag that ToolExecutor will look for and pass as "tag"
        return "product_search"

    async def execute(self, content: str, **kwargs) -> str:
        data = _parse_json_or_wrap(content)
        query: str = str(data.get("query", "")).strip()
        domain: Optional[str] = data.get("domain")
        if isinstance(domain, str):
            domain = domain or None
        try:
            k = int(data.get("k", 5))
        except Exception:
            k = 5

        scored: List[Any] = []
        for item in CATALOG:
            if domain and item.get("domain") != domain:
                continue
            s = _simple_match_score(query, item.get("name", ""))
            if s > 0:
                scored.append((s, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [it for _, it in scored[:k]]
        out = {
            "query": query,
            "domain": domain,
            "results": results,
            "count": len(results),
        }
        return json.dumps(out, ensure_ascii=False)


class InventoryCheckTool(BaseTool):
    """Tag-based inventory check: <inventory_check>...</inventory_check>."""

    @property
    def name(self) -> str:
        return "inventory_check"

    @property
    def trigger_tag(self) -> str:
        return "inventory_check"

    async def execute(self, content: str, **kwargs) -> str:
        data = _parse_json_or_wrap(content)
        sku = data.get("sku")
        product_name = data.get("product_name")
        if sku is not None:
            sku = str(sku)
        if product_name is not None:
            product_name = str(product_name)

        seed = hash((sku or "", product_name or "")) % (10**6)
        rng = random.Random(seed)
        in_stock = rng.random() > 0.15
        qty = rng.randint(1, 50) if in_stock else 0
        restock_days = rng.randint(3, 14) if not in_stock else 0
        out = {
            "sku": sku,
            "product_name": product_name,
            "in_stock": in_stock,
            "quantity": qty,
            "restock_days": restock_days,
        }
        return json.dumps(out, ensure_ascii=False)


class PolicySearchTool(BaseTool):
    """Tag-based policy search: <policy_search>...</policy_search>."""

    @property
    def name(self) -> str:
        return "policy_search"

    @property
    def trigger_tag(self) -> str:
        return "policy_search"

    async def execute(self, content: str, **kwargs) -> str:
        data = _parse_json_or_wrap(content)
        topic = str(data.get("topic", data.get("query", "")) or "")
        topic_l = topic.lower()
        key = "returns"
        if "ship" in topic_l or "delivery" in topic_l or "address" in topic_l:
            key = "shipping"
        elif "warranty" in topic_l or "repair" in topic_l:
            key = "warranty"
        elif "price" in topic_l or "match" in topic_l:
            key = "price_match"
        elif "coupon" in topic_l or "promo" in topic_l or "discount" in topic_l:
            key = "coupons"
        out = {
            "topic": topic,
            "matched_policy": key,
            "policy_text": POLICIES[key],
        }
        return json.dumps(out, ensure_ascii=False)


class OrderLookupTool(BaseTool):
    """Tag-based order lookup: <order_lookup>...</order_lookup>."""

    @property
    def name(self) -> str:
        return "order_lookup"

    @property
    def trigger_tag(self) -> str:
        return "order_lookup"

    async def execute(self, content: str, **kwargs) -> str:
        data = _parse_json_or_wrap(content)
        order_id = str(data.get("order_id", "") or "")

        seed = hash(order_id) % (10**6)
        rng = random.Random(seed)
        item = rng.choice(CATALOG)
        status = rng.choice(["processing", "shipped", "delivered", "refunded", "cancelled"])
        days_ago = rng.randint(1, 45)
        tracking = f"TRK-{rng.randint(1000000, 9999999)}" if status in ["shipped", "delivered"] else None
        out = {
            "order_id": order_id,
            "status": status,
            "item": item,
            "purchase_days_ago": days_ago,
            "tracking_number": tracking,
            "can_change_address": status == "processing",
        }
        return json.dumps(out, ensure_ascii=False)


class PricingCalcTool(BaseTool):
    """Tag-based pricing calculator: <pricing_calc>...</pricing_calc>."""

    @property
    def name(self) -> str:
        return "pricing_calc"

    @property
    def trigger_tag(self) -> str:
        return "pricing_calc"

    async def execute(self, content: str, **kwargs) -> str:
        data = _parse_json_or_wrap(content)
        sku = data.get("sku")
        if sku is not None:
            sku = str(sku)
        unit_price_raw = data.get("unit_price")
        try:
            unit_price: Optional[float] = float(unit_price_raw) if unit_price_raw is not None else None
        except Exception:
            unit_price = None
        try:
            quantity = int(data.get("quantity", 1))
        except Exception:
            quantity = 1
        coupon_code = data.get("coupon_code")
        if coupon_code is not None:
            coupon_code = str(coupon_code)
        shipping_fee_raw = data.get("shipping_fee", 6.99)
        try:
            shipping_fee = float(shipping_fee_raw)
        except Exception:
            shipping_fee = 6.99

        if quantity < 1:
            quantity = 1

        if unit_price is None and sku:
            found = next((x for x in CATALOG if x["sku"] == sku), None)
            if found:
                unit_price = float(found["price"])

        if unit_price is None:
            unit_price = 0.0

        subtotal = float(unit_price) * int(quantity)
        discount = 0.0
        coupon_applied = None
        shipping_final = float(shipping_fee)

        if coupon_code:
            rule = COUPONS.get(coupon_code.strip().upper())
            if rule and subtotal >= float(rule["min_subtotal"]):
                coupon_applied = coupon_code.strip().upper()
                if rule["type"] == "percent":
                    discount = subtotal * (float(rule["value"]) / 100.0)
                elif rule["type"] == "fixed":
                    discount = float(rule["value"])
                elif rule["type"] == "shipping":
                    shipping_final = 0.0

        total = max(0.0, subtotal - discount) + shipping_final
        out = {
            "sku": sku,
            "unit_price": unit_price,
            "quantity": quantity,
            "subtotal": round(subtotal, 2),
            "coupon_code": coupon_code,
            "coupon_applied": coupon_applied,
            "discount": round(discount, 2),
            "shipping_fee": round(shipping_fee, 2),
            "shipping_final": round(shipping_final, 2),
            "total": round(total, 2),
            "currency": "USD",
        }
        return json.dumps(out, ensure_ascii=False)


__all__ = [
    "POLICIES",
    "CATALOG",
    "COUPONS",
    "ProductSearchTool",
    "InventoryCheckTool",
    "PolicySearchTool",
    "OrderLookupTool",
    "PricingCalcTool",
]

