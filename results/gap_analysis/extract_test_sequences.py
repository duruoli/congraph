import json
import csv
import os

base_dir = '/Users/duruoli/A/A李杜若/1-科研/PhD/0/1-code/congraph/results'
oracle_file = os.path.join(base_dir, 'test_seq_comparison', 'rubric_sim_oracle.json')

diseases = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']

data_rows = []

print("Loading oracle data...")
with open(oracle_file, 'r') as f:
    oracle_data = json.load(f)

for disease in diseases:
    feature_file = os.path.join(base_dir, f'{disease}_features.json')
    if not os.path.exists(feature_file):
        print(f"Warning: {feature_file} not found.")
        continue
        
    print(f"Loading {disease} features...")
    with open(feature_file, 'r') as f:
        feature_data = json.load(f)
        
    sim_patients = oracle_data.get('results', {}).get(disease, {})
    actual_patients = feature_data.get('results', {})
    
    for pat_id, pat_sim_data in sim_patients.items():
        sim_seq = pat_sim_data.get('test_sequence', [])
        sim_seq = ['Lab_Panel'] + sim_seq
        
        # Get actual tests_done
        actual_seq = []
        if pat_id in actual_patients:
            pat_actual_steps = actual_patients[pat_id]
            if pat_actual_steps:
                last_step = pat_actual_steps[-1]
                actual_seq = last_step.get('features', {}).get('tests_done', [])
        
        data_rows.append({
            'patient_id': pat_id,
            'disease': disease,
            'simulated_sequence': ", ".join(sim_seq),
            'actual_sequence': ", ".join(actual_seq),
            'simulated_length': len(sim_seq),
            'actual_length': len(actual_seq)
        })

output_csv = os.path.join(base_dir, 'test_seq_comparison', 'test_sequence_comparison.csv')
print(f"Writing output to {output_csv}...")
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['patient_id', 'disease', 'simulated_sequence', 'actual_sequence', 'simulated_length', 'actual_length'])
    writer.writeheader()
    writer.writerows(data_rows)

print(f"Successfully wrote {len(data_rows)} rows to {output_csv}")
