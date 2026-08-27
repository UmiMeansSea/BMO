import csv

with open(r'../data/bmo_french_dataset.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

total = len(rows)
erroneous = sum(1 for r in rows if r['Is_Erroneous'] == 'TRUE')
correct = sum(1 for r in rows if r['Is_Erroneous'] == 'FALSE')

error_types = {}
for r in rows:
    et = r['Error_Type']
    error_types[et] = error_types.get(et, 0) + 1

cefr = {}
for r in rows:
    c = r['CEFR_Level']
    cefr[c] = cefr.get(c, 0) + 1

print(f"Total rows:              {total}")
print(f"Erroneous prompts (TRUE): {erroneous}")
print(f"Correct prompts (FALSE):  {correct}")
print()
print("Error type distribution:")
for k, v in sorted(error_types.items()):
    print(f"  {k}: {v}")
print()
print("CEFR level distribution:")
for k, v in sorted(cefr.items()):
    print(f"  {k}: {v}")
print()
print("Column headers:", list(rows[0].keys()))
print()
print("Sample erroneous row:")
for r in rows:
    if r['Is_Erroneous'] == 'TRUE':
        print("  ID:    " + r['Prompt_ID'])
        print("  Input: " + r['User_Input_French'])
        print("  Type:  " + r['Error_Type'])
        print("  Fix:   " + r['Expected_Correction'])
        print("  Rule:  " + r['Error_Subtype'])
        break
