"""Seed data for ConstructIQ.

SYNTHETIC DEMO DATA — every vendor, quote and requirement in this file is
fictional, generated for a hackathon prototype. This is NOT a real vendor
directory; names, contacts, prices and performance figures are invented
(though kept plausible for the Indian construction-materials market:
IS 1786 TMT grades Fe415/Fe500/Fe500D, OPC 33/43/53 & PPC cement,
ready-mix concrete M20–M60, aggregates, curtain-wall glass, HVAC units).

Idempotent: ``seed_all(db_session)`` skips any table that already has rows.
Runnable directly: ``python -m backend.db.seed_vendors``.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import date

from sqlalchemy.orm import Session

from backend.db.models import ExtractedRequirement, Vendor, VendorQuote

logger = logging.getLogger(__name__)

DEMO_PROJECT_ID = "PRJ-2024-001"
DEMO_SOURCE_FILE = "demo_seed"

# Plausible INR market base rates per material (rate, unit, grade|None).
# Quote prices are derived from these, scaled by each vendor's price_index.
BASE_RATES: dict[str, tuple[float, str, str | None]] = {
    "Fe415 TMT steel": (54000.0, "tonne", "Fe415"),
    "Fe500 TMT steel": (56500.0, "tonne", "Fe500"),
    "Fe500D TMT steel": (58500.0, "tonne", "Fe500D"),
    "Structural steel sections": (62000.0, "tonne", "E250"),
    "OPC 33 cement": (340.0, "bag", "OPC 33"),
    "OPC 43 cement": (360.0, "bag", "OPC 43"),
    "OPC 53 cement": (385.0, "bag", "OPC 53"),
    "PPC cement": (350.0, "bag", "PPC"),
    "M20 ready-mix concrete": (4300.0, "cum", "M20"),
    "M25 ready-mix concrete": (4700.0, "cum", "M25"),
    "M30 ready-mix concrete": (5100.0, "cum", "M30"),
    "M40 ready-mix concrete": (5900.0, "cum", "M40"),
    "M50 ready-mix concrete": (6800.0, "cum", "M50"),
    "M60 ready-mix concrete": (7800.0, "cum", "M60"),
    "20mm coarse aggregate": (1350.0, "tonne", None),
    "10mm coarse aggregate": (1400.0, "tonne", None),
    "Manufactured sand (M-sand)": (1150.0, "tonne", None),
    "Double-glazed curtain wall glass": (5800.0, "sqm", None),
    "Toughened glass panels": (3200.0, "sqm", None),
    "VRF HVAC units": (285000.0, "unit", None),
    "Ducted split HVAC units": (145000.0, "unit", None),
    "Air handling units (AHU)": (350000.0, "unit", None),
}

PAYMENT_TERMS_POOL = [
    "Net 30",
    "Net 45",
    "Net 15",
    "50% advance, balance on delivery",
    "30% advance, Net 30 on balance",
]

# 17 fully fictional vendors, deliberately varied across every attribute so
# risk / evaluation / negotiation demos have real spread to work with.
VENDOR_SEED_DATA: list[dict] = [
    {
        "name": "Shree Balaji Steel Traders",
        "materials": ["Fe415 TMT steel", "Fe500 TMT steel", "Fe500D TMT steel"],
        "location": "Mumbai", "region": "West India",
        "contact_email": "sales@shreebalajisteel.example.in",
        "contact_phone": "+91-98211-40021",
        "rating": 4.5, "avg_delivery_days": 7, "historical_on_time_pct": 94.5,
        "typical_order_size": "20–500 tonne consignments",
        "typical_min_qty": 20.0, "typical_max_qty": 500.0,
        "price_index": 1.04, "negotiation_flexibility": 0.35,
    },
    {
        "name": "Deccan Cement Depot",
        "materials": ["OPC 43 cement", "OPC 53 cement", "PPC cement"],
        "location": "Hyderabad", "region": "South India",
        "contact_email": "orders@deccancement.example.in",
        "contact_phone": "+91-90001-27364",
        "rating": 4.2, "avg_delivery_days": 5, "historical_on_time_pct": 91.0,
        "typical_order_size": "200–5,000 bag lots",
        "typical_min_qty": 200.0, "typical_max_qty": 5000.0,
        "price_index": 0.98, "negotiation_flexibility": 0.45,
    },
    {
        "name": "Kanchan TMT & Alloys",
        "materials": ["Fe500 TMT steel", "Fe500D TMT steel", "Structural steel sections"],
        "location": "Raipur", "region": "Central India",
        "contact_email": "kanchan.tmt@example.co.in",
        "contact_phone": "+91-77121-88450",
        "rating": 4.0, "avg_delivery_days": 10, "historical_on_time_pct": 88.0,
        "typical_order_size": "50–1,200 tonne mill despatches",
        "typical_min_qty": 50.0, "typical_max_qty": 1200.0,
        "price_index": 0.95, "negotiation_flexibility": 0.65,
    },
    {
        "name": "Ganga Valley RMC Plant",
        "materials": ["M20 ready-mix concrete", "M25 ready-mix concrete", "M30 ready-mix concrete"],
        "location": "Kanpur", "region": "North India",
        "contact_email": "dispatch@gangavalleyrmc.example.in",
        "contact_phone": "+91-91234-55710",
        "rating": 3.6, "avg_delivery_days": 3, "historical_on_time_pct": 78.0,
        "typical_order_size": "6–400 cum pours per booking",
        "typical_min_qty": 6.0, "typical_max_qty": 400.0,
        "price_index": 0.93, "negotiation_flexibility": 0.5,
    },
    {
        "name": "Marudhar Aggregates & Sand",
        "materials": ["20mm coarse aggregate", "10mm coarse aggregate", "Manufactured sand (M-sand)"],
        "location": "Jodhpur", "region": "West India",
        "contact_email": "marudhar.agg@example.in",
        "contact_phone": "+91-94141-30298",
        "rating": 3.8, "avg_delivery_days": 6, "historical_on_time_pct": 84.0,
        "typical_order_size": "10–800 tonne truckloads",
        "typical_min_qty": 10.0, "typical_max_qty": 800.0,
        "price_index": 0.92, "negotiation_flexibility": 0.7,
    },
    {
        "name": "Nilgiri Glass Facades",
        "materials": ["Double-glazed curtain wall glass", "Toughened glass panels"],
        "location": "Coimbatore", "region": "South India",
        "contact_email": "projects@nilgiriglass.example.in",
        "contact_phone": "+91-98430-61175",
        "rating": 4.4, "avg_delivery_days": 28, "historical_on_time_pct": 90.0,
        "typical_order_size": "100–3,000 sqm facade packages",
        "typical_min_qty": 100.0, "typical_max_qty": 3000.0,
        "price_index": 1.10, "negotiation_flexibility": 0.2,
    },
    {
        "name": "Hindustan Klima HVAC",
        "materials": ["VRF HVAC units", "Air handling units (AHU)", "Ducted split HVAC units"],
        "location": "New Delhi", "region": "North India",
        "contact_email": "enquiry@hindustanklima.example.in",
        "contact_phone": "+91-98100-74412",
        "rating": 4.6, "avg_delivery_days": 35, "historical_on_time_pct": 93.0,
        "typical_order_size": "2–60 units per project order",
        "typical_min_qty": 2.0, "typical_max_qty": 60.0,
        "price_index": 1.12, "negotiation_flexibility": 0.18,
    },
    {
        "name": "Sardar Steel Syndicate",
        "materials": ["Fe415 TMT steel", "Fe500 TMT steel"],
        "location": "Ludhiana", "region": "North India",
        "contact_email": "sardarsteel@example.co.in",
        "contact_phone": "+91-98761-20934",
        "rating": 3.2, "avg_delivery_days": 12, "historical_on_time_pct": 72.0,
        "typical_order_size": "10–300 tonne spot lots",
        "typical_min_qty": 10.0, "typical_max_qty": 300.0,
        "price_index": 0.94, "negotiation_flexibility": 0.75,
    },
    {
        "name": "Konark Cement Agencies",
        "materials": ["OPC 33 cement", "OPC 43 cement", "PPC cement"],
        "location": "Bhubaneswar", "region": "East India",
        "contact_email": "konarkcement@example.in",
        "contact_phone": "+91-94370-18852",
        "rating": 3.9, "avg_delivery_days": 8, "historical_on_time_pct": 86.0,
        "typical_order_size": "100–4,000 bag despatches",
        "typical_min_qty": 100.0, "typical_max_qty": 4000.0,
        "price_index": 0.96, "negotiation_flexibility": 0.55,
    },
    {
        "name": "Vindhya RMC & Infra",
        "materials": ["M30 ready-mix concrete", "M40 ready-mix concrete", "M50 ready-mix concrete"],
        "location": "Bhopal", "region": "Central India",
        "contact_email": "plant@vindhyarmc.example.in",
        "contact_phone": "+91-75524-40318",
        "rating": 4.3, "avg_delivery_days": 4, "historical_on_time_pct": 92.0,
        "typical_order_size": "12–600 cum scheduled pours",
        "typical_min_qty": 12.0, "typical_max_qty": 600.0,
        "price_index": 1.02, "negotiation_flexibility": 0.3,
    },
    {
        "name": "Charminar Structural Steels",
        "materials": ["Structural steel sections", "Fe500D TMT steel"],
        "location": "Hyderabad", "region": "South India",
        "contact_email": "sales@charminarsteels.example.in",
        "contact_phone": "+91-90309-55127",
        "rating": 4.7, "avg_delivery_days": 14, "historical_on_time_pct": 95.0,
        "typical_order_size": "40–900 tonne fabrication-grade lots",
        "typical_min_qty": 40.0, "typical_max_qty": 900.0,
        "price_index": 1.08, "negotiation_flexibility": 0.12,
    },
    {
        "name": "Aravalli Stone Crushers",
        "materials": ["20mm coarse aggregate", "10mm coarse aggregate"],
        "location": "Udaipur", "region": "West India",
        "contact_email": "aravallicrushers@example.in",
        "contact_phone": "+91-94143-77209",
        "rating": 3.0, "avg_delivery_days": 9, "historical_on_time_pct": 68.0,
        "typical_order_size": "25–1,500 tonne bulk supply",
        "typical_min_qty": 25.0, "typical_max_qty": 1500.0,
        "price_index": 0.92, "negotiation_flexibility": 0.8,
    },
    {
        "name": "Brahmaputra Glass & Glazing",
        "materials": ["Double-glazed curtain wall glass", "Toughened glass panels"],
        "location": "Guwahati", "region": "East India",
        "contact_email": "works@brahmaputraglazing.example.in",
        "contact_phone": "+91-98640-31586",
        "rating": 3.4, "avg_delivery_days": 45, "historical_on_time_pct": 74.0,
        "typical_order_size": "50–1,500 sqm glazing lots",
        "typical_min_qty": 50.0, "typical_max_qty": 1500.0,
        "price_index": 0.97, "negotiation_flexibility": 0.4,
    },
    {
        "name": "Coromandel Cooling Systems",
        "materials": ["VRF HVAC units", "Ducted split HVAC units"],
        "location": "Chennai", "region": "South India",
        "contact_email": "sales@coromandelcooling.example.in",
        "contact_phone": "+91-98410-92273",
        "rating": 4.1, "avg_delivery_days": 30, "historical_on_time_pct": 89.0,
        "typical_order_size": "1–40 units per order",
        "typical_min_qty": 1.0, "typical_max_qty": 40.0,
        "price_index": 1.05, "negotiation_flexibility": 0.25,
    },
    {
        "name": "Shivalik Concrete Works",
        "materials": ["M25 ready-mix concrete", "M40 ready-mix concrete", "M60 ready-mix concrete"],
        "location": "Chandigarh", "region": "North India",
        "contact_email": "orders@shivalikconcrete.example.in",
        "contact_phone": "+91-98760-14405",
        "rating": 4.9, "avg_delivery_days": 5, "historical_on_time_pct": 97.0,
        "typical_order_size": "20–800 cum premium pours",
        "typical_min_qty": 20.0, "typical_max_qty": 800.0,
        "price_index": 1.15, "negotiation_flexibility": 0.05,
    },
    {
        "name": "Mahalaxmi Build Mart",
        "materials": ["OPC 53 cement", "PPC cement", "Fe500 TMT steel"],
        "location": "Pune", "region": "West India",
        "contact_email": "mahalaxmi.buildmart@example.in",
        "contact_phone": "+91-98220-63741",
        "rating": 3.7, "avg_delivery_days": 11, "historical_on_time_pct": 82.0,
        "typical_order_size": "50–2,000 bag orders (small steel lots too)",
        "typical_min_qty": 50.0, "typical_max_qty": 2000.0,
        "price_index": 1.00, "negotiation_flexibility": 0.5,
    },
    {
        "name": "Godavari Steel & Cement Corp",
        "materials": ["Fe500D TMT steel", "OPC 53 cement"],
        "location": "Nagpur", "region": "Central India",
        "contact_email": "corp@godavaristeelcement.example.in",
        "contact_phone": "+91-71226-90513",
        "rating": 4.8, "avg_delivery_days": 6, "historical_on_time_pct": 98.0,
        "typical_order_size": "30–700 tonne steel consignments",
        "typical_min_qty": 30.0, "typical_max_qty": 700.0,
        "price_index": 1.06, "negotiation_flexibility": 0.22,
    },
]


def seed_vendors(db_session: Session) -> int:
    """Insert the fictional vendor directory. Skips (returns 0) if any
    vendors already exist."""
    existing = db_session.query(Vendor).count()
    if existing:
        logger.info("seed_vendors: %d vendors already present, skipping", existing)
        return 0

    for spec in VENDOR_SEED_DATA:
        db_session.add(
            Vendor(
                name=spec["name"],
                materials_supplied=json.dumps(spec["materials"]),
                location=spec["location"],
                region=spec["region"],
                contact_email=spec["contact_email"],
                contact_phone=spec["contact_phone"],
                rating=spec["rating"],
                avg_delivery_days=spec["avg_delivery_days"],
                historical_on_time_pct=spec["historical_on_time_pct"],
                typical_order_size=spec["typical_order_size"],
                typical_min_qty=spec["typical_min_qty"],
                typical_max_qty=spec["typical_max_qty"],
                price_index=spec["price_index"],
                negotiation_flexibility=spec["negotiation_flexibility"],
            )
        )
    db_session.commit()
    count = len(VENDOR_SEED_DATA)
    logger.info("seed_vendors: inserted %d synthetic vendors", count)
    return count


def seed_quotes(db_session: Session) -> int:
    """Insert 2-3 VendorQuote rows per vendor for materials that vendor
    actually supplies. Prices = plausible INR base rate x vendor price_index
    with small deterministic jitter (fixed random.Random(42)).
    Skips (returns 0) if any quotes already exist."""
    existing = db_session.query(VendorQuote).count()
    if existing:
        logger.info("seed_quotes: %d quotes already present, skipping", existing)
        return 0

    vendors = db_session.query(Vendor).order_by(Vendor.id).all()
    if not vendors:
        logger.warning("seed_quotes: no vendors in DB, nothing to quote")
        return 0

    rng = random.Random(42)
    count = 0
    for vendor in vendors:
        materials = [m for m in vendor.materials_list() if m in BASE_RATES]
        quoted_materials = materials[:3]  # 2-3 per vendor by construction
        for material in quoted_materials:
            base_rate, unit, grade = BASE_RATES[material]
            jitter = rng.uniform(-0.03, 0.03)
            quoted_price = round(base_rate * vendor.price_index * (1 + jitter), 2)
            quantity = float(round(rng.uniform(vendor.typical_min_qty, vendor.typical_max_qty)))
            delivery_days = max(1, vendor.avg_delivery_days + rng.randint(-2, 3))
            db_session.add(
                VendorQuote(
                    project_id=DEMO_PROJECT_ID,
                    vendor_id=vendor.id,
                    material=material,
                    grade=grade,
                    quoted_price=quoted_price,
                    quantity=quantity,
                    unit=unit,
                    delivery_days=delivery_days,
                    payment_terms=rng.choice(PAYMENT_TERMS_POOL),
                    valid_until=date(2026, 9, 30),
                )
            )
            count += 1
    db_session.commit()
    logger.info("seed_quotes: inserted %d synthetic quotes", count)
    return count


def seed_demo_requirements(db_session: Session) -> int:
    """Insert the 4 demo ExtractedRequirement rows for PRJ-2024-001
    (source_file="demo_seed"). Skips (returns 0) if they already exist."""
    existing = (
        db_session.query(ExtractedRequirement)
        .filter(
            ExtractedRequirement.project_id == DEMO_PROJECT_ID,
            ExtractedRequirement.source_file == DEMO_SOURCE_FILE,
        )
        .count()
    )
    if existing:
        logger.info("seed_demo_requirements: %d rows already present, skipping", existing)
        return 0

    rows = [
        ExtractedRequirement(
            project_id=DEMO_PROJECT_ID,
            source_file=DEMO_SOURCE_FILE,
            material="TMT steel",
            grade="Fe500D",
            quantity=120.0,
            unit="tonne",
            deadline="2026-08-10",
            certification="IS 1786",
            source_page=0,
            confidence="high",
        ),
        ExtractedRequirement(
            project_id=DEMO_PROJECT_ID,
            source_file=DEMO_SOURCE_FILE,
            material="OPC 53 cement",
            grade="OPC 53",
            quantity=800.0,
            unit="bag",
            deadline="2026-07-30",
            certification="IS 269",
            source_page=0,
            confidence="high",
        ),
        ExtractedRequirement(
            project_id=DEMO_PROJECT_ID,
            source_file=DEMO_SOURCE_FILE,
            material="Ready-mix concrete",
            grade="M40",
            quantity=350.0,
            unit="cum",
            deadline="2026-09-01",
            certification="IS 4926",
            source_page=0,
            confidence="high",
        ),
        ExtractedRequirement(
            project_id=DEMO_PROJECT_ID,
            source_file=DEMO_SOURCE_FILE,
            material="Double-glazed curtain wall glass",
            grade=None,
            quantity=1200.0,
            unit="sqm",
            deadline="2026-10-15",
            certification=None,
            source_page=0,
            confidence="high",
        ),
    ]
    db_session.add_all(rows)
    db_session.commit()
    logger.info("seed_demo_requirements: inserted %d demo requirements", len(rows))
    return len(rows)


def seed_all(db_session: Session) -> dict:
    """Run every seeder (each is individually idempotent).

    Returns {"vendors": n, "quotes": n, "requirements": n} — counts of rows
    actually inserted this call (all zero on a repeat run)."""
    counts = {
        "vendors": seed_vendors(db_session),
        "quotes": seed_quotes(db_session),
        "requirements": seed_demo_requirements(db_session),
    }
    logger.info("seed_all: %s", counts)
    return counts


if __name__ == "__main__":
    import sys

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend import config
    from backend.db.models import Base

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    engine = create_engine(
        f"sqlite:///{config.DB_PATH}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with session_factory() as session:
        summary = seed_all(session)
    logger.info(
        "Seed complete at %s — vendors=%d quotes=%d requirements=%d",
        config.DB_PATH,
        summary["vendors"],
        summary["quotes"],
        summary["requirements"],
    )
