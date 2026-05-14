import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.test_burden_cost import lab_panel_cost_usd, TEST_COST

def evaluate_patient(disease, patient_id):
    path = REPO_ROOT / f"results/{disease}_features.json"
    with path.open() as f:
        data = json.load(f)
        
    patient_data = data["results"].get(patient_id)
    if not patient_data:
        print(f"Patient {patient_id} not found in {disease}.")
        return

    # To get total cost, we look at the trajectory tests and lab items
    # In the JSON, the final step will have all cumulative tests_done and lab_itemids
    final_step = patient_data[-1]
    features = final_step.get("features", {})
    tests_done = features.get("tests_done", [])
    lab_itemids = features.get("lab_itemids", [])

    print(f"--- Patient {patient_id} ({disease}) ---")
    
    # Calculate lab cost
    l_cost = lab_panel_cost_usd(lab_itemids)
    print(f"Lab Items Count: {len(lab_itemids)}")
    print(f"Lab Items: {', '.join(lab_itemids)}")
    print(f"  -> Total Lab Cost: ${l_cost:.2f}")

    # Calculate imaging cost
    total_img_cost = 0.0
    valid_imaging = []
    
    # Wait, tests_done usually only contains things like "CT_Abdomen", "Radiograph_Chest", etc.
    for test in tests_done:
        if test == "Lab_Panel": continue
        c = TEST_COST.get(test, 0.0)
        total_img_cost += c
        valid_imaging.append(f"{test} (${c:.2f})")
    
    print(f"Imaging Tests: {', '.join(valid_imaging) if valid_imaging else 'None'}")
    print(f"  -> Total Imaging Cost: ${total_img_cost:.2f}")

    total_cost = l_cost + total_img_cost
    print(f"TOTAL DIAGNOSTIC COST: ${total_cost:.2f}\n")


evaluate_patient("appendicitis", "20890008")
evaluate_patient("cholecystitis", "20000019")
evaluate_patient("pancreatitis", "20005119")
