import pandas as pd
import numpy as np
import json
import time

start_time = time.time()
print("Starting deep inspection...")

# 1. Inspect users_data
df_users = pd.read_csv('users_data.csv')
print(f"Users shape: {df_users.shape}")
user_ids = set(df_users['id'])

# 2. Inspect transactions_data in chunks
total_rows = 0
min_date = None
max_date = None
unique_clients = set()
use_chip_counts = {}
mcc_counts = {}
errors_counts = {}
missing_counts = {}
amount_stats = {'pos': 0, 'zero': 0, 'neg': 0, 'total': 0, 'sum_pos': 0.0, 'sum_neg': 0.0}

chunksize = 500_000
chunk_idx = 0

for chunk in pd.read_csv('transactions_data.csv', chunksize=chunksize, 
                         dtype={'id': 'int64', 'client_id': 'int64', 'card_id': 'int64', 
                                'amount': 'str', 'use_chip': 'str', 'merchant_id': 'int64',
                                'merchant_city': 'str', 'merchant_state': 'str', 
                                'zip': 'float64', 'mcc': 'int64', 'errors': 'str'}):
    chunk_idx += 1
    total_rows += len(chunk)
    
    # Dates
    c_min = chunk['date'].min()
    c_max = chunk['date'].max()
    if min_date is None or c_min < min_date:
        min_date = c_min
    if max_date is None or c_max > max_date:
        max_date = c_max
        
    # Unique clients
    unique_clients.update(chunk['client_id'].unique())
    
    # use_chip
    vc_chip = chunk['use_chip'].value_counts(dropna=False)
    for k, v in vc_chip.items():
        use_chip_counts[str(k)] = use_chip_counts.get(str(k), 0) + v
        
    # mcc
    vc_mcc = chunk['mcc'].value_counts()
    for k, v in vc_mcc.items():
        mcc_counts[k] = mcc_counts.get(k, 0) + v
        
    # errors
    vc_err = chunk['errors'].value_counts(dropna=False)
    for k, v in vc_err.items():
        errors_counts[str(k)] = errors_counts.get(str(k), 0) + v
        
    # Missing values
    for col in chunk.columns:
        missing_counts[col] = missing_counts.get(col, 0) + chunk[col].isnull().sum()
        
    # Amount numeric check
    amt_clean = chunk['amount'].str.replace('$', '', regex=False).str.strip().astype(float)
    pos_mask = amt_clean > 0
    neg_mask = amt_clean < 0
    zero_mask = amt_clean == 0
    
    amount_stats['pos'] += pos_mask.sum()
    amount_stats['neg'] += neg_mask.sum()
    amount_stats['zero'] += zero_mask.sum()
    amount_stats['sum_pos'] += amt_clean[pos_mask].sum()
    amount_stats['sum_neg'] += amt_clean[neg_mask].sum()

print(f"Inspection completed in {time.time() - start_time:.2f} seconds.")
print(f"Total transaction rows: {total_rows}")
print(f"Date range: {min_date} to {max_date}")
print(f"Unique clients in transactions: {len(unique_clients)}")
print(f"Unique clients in users: {len(user_ids)}")
print(f"Intersection of clients: {len(unique_clients.intersection(user_ids))}")
print(f"Users without transactions: {len(user_ids - unique_clients)}")
print(f"Tx clients not in users: {len(unique_clients - user_ids)}")

print("\nuse_chip distribution:")
for k, v in use_chip_counts.items():
    print(f"  {k}: {v} ({v/total_rows*100:.2f}%)")

print("\nTop 30 MCC codes:")
sorted_mcc = sorted(mcc_counts.items(), key=lambda x: x[1], reverse=True)
for k, v in sorted_mcc[:30]:
    print(f"  MCC {k}: {v} ({v/total_rows*100:.2f}%)")

print("\nErrors distribution:")
for k, v in errors_counts.items():
    print(f"  {k}: {v}")

print("\nMissing values:")
for k, v in missing_counts.items():
    print(f"  {k}: {v}")

print("\nAmount breakdown:")
print(f"  Positive: {amount_stats['pos']} (sum: ${amount_stats['sum_pos']:,.2f})")
print(f"  Negative (refunds): {amount_stats['neg']} (sum: ${amount_stats['sum_neg']:,.2f})")
print(f"  Zero: {amount_stats['zero']}")

# Check if 6011 (ATM) exists
print(f"\nDoes MCC 6011 exist in MCCs? {6011 in mcc_counts} (count: {mcc_counts.get(6011, 0)})")
# Check cash disbursement codes (6010, 6011, 6012)
for cash_mcc in [6010, 6011, 6012]:
    print(f"  MCC {cash_mcc}: {mcc_counts.get(cash_mcc, 0)}")

# Save inspection summary to json
with open('inspection_summary.json', 'w') as f:
    json.dump({
        'total_rows': int(total_rows),
        'min_date': str(min_date),
        'max_date': str(max_date),
        'unique_clients_tx': len(unique_clients),
        'unique_users': len(user_ids),
        'overlap': len(unique_clients.intersection(user_ids)),
        'use_chip_counts': {str(k): int(v) for k, v in use_chip_counts.items()},
        'top_30_mcc': [{'mcc': int(k), 'count': int(v)} for k, v in sorted_mcc[:30]],
        'errors_counts': {str(k): int(v) for k, v in errors_counts.items()},
        'amount_stats': {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in amount_stats.items()}
    }, f, indent=2)
print("Saved inspection_summary.json")
