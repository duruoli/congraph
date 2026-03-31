import csv, json, sys

samples = {'Ultrasound': [], 'CT': [], 'Radiograph': [], 'MRI': []}
max_per = 8

for disease in ['cholecystitis', 'appendicitis', 'pancreatitis', 'diverticulitis']:
    with open(f'raw_data/{disease}_hadm_info_first_diag.csv') as f:
        for row in csv.DictReader(f):
            try:
                rads = json.loads(row.get('Radiology', '[]') or '[]')
            except Exception:
                continue
            for r in rads:
                mod = r.get('Modality', '')
                if mod in samples and len(samples[mod]) < max_per:
                    samples[mod].append({
                        'disease': disease,
                        'exam': r.get('Exam Name', ''),
                        'report': r.get('Report', ''),
                    })

for mod, reports in samples.items():
    print('\n' + '=' * 72)
    print('MODALITY: %s  (%d samples)' % (mod, len(reports)))
    for i, r in enumerate(reports[:4]):
        print('\n  [%d] disease=%s  exam=%s' % (i + 1, r['disease'], r['exam']))
        text = r['report'][:800]
        for line in text.split('\n'):
            print('  ' + line)
