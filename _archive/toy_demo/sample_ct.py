import csv, json, sys

targets = {
    'appendicitis_CT': [],
    'pancreatitis_CT': [],
    'diverticulitis_CT': [],
}

for disease in ['appendicitis', 'pancreatitis', 'diverticulitis']:
    key = disease + '_CT'
    with open('data/raw_data/%s_hadm_info_first_diag.csv' % disease) as f:
        for row in csv.DictReader(f):
            try:
                rads = json.loads(row.get('Radiology', '[]') or '[]')
            except Exception:
                continue
            for r in rads:
                if r.get('Modality', '') == 'CT' and len(targets[key]) < 4:
                    targets[key].append(r.get('Report', ''))

for label, reports in targets.items():
    print('\n' + '=' * 72)
    print('CATEGORY: ' + label)
    for i, rep in enumerate(reports):
        print('\n  [%d]' % (i + 1))
        text = rep
        start = text.find('FINDINGS')
        if start > 0:
            text = text[start:]
        for line in text[:900].split('\n'):
            sys.stdout.write('  ' + line + '\n')
