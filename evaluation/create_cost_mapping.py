import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Pull existing constants from test_burden_cost
from evaluation.test_burden_cost import (
    TEST_COST,
    LAB_COMPONENT_USD,
    UNMAPPED_LAB_ITEM_USD,
    _CBC_ITEMIDS,
    _CMP_ITEMIDS,
    _URINALYSIS_ITEMIDS,
    _LIPASE_ITEMIDS,
    _CRP_ITEMIDS,
    _HCG_ITEMIDS,
)

CSV_PATH = REPO_ROOT / "data" / "raw_data" / "cost_mapping.csv"

def main():
    rows = []
    
    # 1. Imaging
    imaging_cpt = {
        "Radiograph_Chest": "71046",
        "Ultrasound_Abdomen": "76705",
        "HIDA_Scan": "78226",
        "CT_Abdomen": "74177",
        "MRCP_Abdomen": "74183",
        "MRI_Abdomen": "74183",
    }
    
    for mod, cost in TEST_COST.items():
        if mod == "Lab_Panel":
            continue
        rows.append({
            "item_type": "imaging",
            "item_id": mod,
            "category": "VALID_TESTS",
            "cpt_code": imaging_cpt.get(mod, ""),
            "cost_usd": cost,
            "description": f"{mod} baseline cost",
        })

    # 2. Lab Bundles
    bundle_cpt = {
        "cbc": "85025",
        "cmp": "80053",
        "lipase": "83690",
        "urinalysis": "81003",
        "crp": "86140",
        "hcg": "84703",
    }
    
    for comp, cost in LAB_COMPONENT_USD.items():
        rows.append({
            "item_type": "lab_bundle",
            "item_id": comp,
            "category": "Blood/Urine",
            "cpt_code": bundle_cpt.get(comp, ""),
            "cost_usd": cost,
            "description": f"{comp.upper()} bundle",
        })

    # 3. Lab Item IDs (Mapping itemid -> bundle)
    # Note: cost_usd is empty because the cost is at the bundle level
    # We will use this to replace the hardcoded frozensets in test_burden_cost.py
    bundles_to_ids = {
        "cbc": _CBC_ITEMIDS,
        "cmp": _CMP_ITEMIDS,
        "urinalysis": _URINALYSIS_ITEMIDS,
        "lipase": _LIPASE_ITEMIDS,
        "crp": _CRP_ITEMIDS,
        "hcg": _HCG_ITEMIDS,
    }
    
    for comp, ids in bundles_to_ids.items():
        for iid in sorted(ids, key=lambda x: int(x)):
            rows.append({
                "item_type": "lab_itemid",
                "item_id": str(iid),
                "category": comp,  # category points to the bundle
                "cpt_code": "",
                "cost_usd": "",
                "description": f"MIMIC-IV itemid mapped to {comp}",
            })

    # 4. Fees / Defaults
    rows.append({
        "item_type": "fee",
        "item_id": "venipuncture",
        "category": "Procedure",
        "cpt_code": "36415",
        "cost_usd": 8.83,
        "description": "Venipuncture flat fee for blood tests",
    })
    rows.append({
        "item_type": "fee",
        "item_id": "unmapped_lab",
        "category": "Fallback",
        "cpt_code": "",
        "cost_usd": UNMAPPED_LAB_ITEM_USD,
        "description": "Marginal a la carte cost for unmapped itemids",
    })

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_type", "item_id", "category", "cpt_code", "cost_usd", "description"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Created {CSV_PATH} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
