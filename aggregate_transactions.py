import pandas as pd
import numpy as np
import time
from collections import defaultdict, Counter

print("Starting high-performance vectorized aggregation script...")
start_time = time.time()

BILL_MCCS = {4900, 4814, 4899, 6300}

# Aggregation accumulators
client_txn_total = Counter()
client_online = Counter()
client_swipe = Counter()
client_chip = Counter()

client_pos_count = Counter()
client_pos_sum = defaultdict(float)
client_pos_amounts = defaultdict(list)

client_refund_count = Counter()
client_refund_sum = defaultdict(float)
client_zero_count = Counter()

client_error_count = Counter()
client_bill_count = Counter()

client_min_date = {}
client_max_date = {}
client_year_months = defaultdict(set)
client_states = defaultdict(Counter)

chunksize = 1_000_000
chunk_num = 0

for chunk in pd.read_csv('transactions_data.csv', chunksize=chunksize,
                         usecols=['client_id', 'date', 'amount', 'use_chip', 'merchant_state', 'mcc', 'errors'],
                         dtype={'client_id': 'int64', 'date': 'str', 'amount': 'str', 
                                'use_chip': 'str', 'merchant_state': 'str', 'mcc': 'int64', 'errors': 'str'}):
    chunk_num += 1
    t0 = time.time()
    
    # Amount numeric
    amt = chunk['amount'].str.replace('$', '', regex=False).str.strip().astype(np.float32)
    chunk['amt'] = amt
    cid = chunk['client_id']
    
    # 1. Counts per client
    for c, cnt in cid.value_counts().items():
        client_txn_total[c] += cnt
        
    # 2. Chip counts
    for c, cnt in cid[chunk['use_chip'] == 'Online Transaction'].value_counts().items():
        client_online[c] += cnt
    for c, cnt in cid[chunk['use_chip'] == 'Swipe Transaction'].value_counts().items():
        client_swipe[c] += cnt
    for c, cnt in cid[chunk['use_chip'] == 'Chip Transaction'].value_counts().items():
        client_chip[c] += cnt
        
    # 3. Positive amounts
    pos_mask = chunk['amt'] > 0
    if pos_mask.any():
        pos_df = chunk.loc[pos_mask, ['client_id', 'amt']]
        grp_pos = pos_df.groupby('client_id')['amt']
        for c, cnt in grp_pos.count().items():
            client_pos_count[c] += cnt
        for c, s in grp_pos.sum().items():
            client_pos_sum[c] += s
        # Store amounts for median
        for c, vals in pos_df.groupby('client_id')['amt']:
            client_pos_amounts[c].extend(vals.tolist())
            
    # 4. Negative amounts (refunds)
    neg_mask = chunk['amt'] < 0
    if neg_mask.any():
        neg_df = chunk.loc[neg_mask, ['client_id', 'amt']]
        grp_neg = neg_df.groupby('client_id')['amt']
        for c, cnt in grp_neg.count().items():
            client_refund_count[c] += cnt
        for c, s in grp_neg.sum().items():
            client_refund_sum[c] += s
            
    # 5. Zero amounts
    zero_mask = chunk['amt'] == 0
    if zero_mask.any():
        for c, cnt in cid[zero_mask].value_counts().items():
            client_zero_count[c] += cnt
            
    # 6. Errors
    err_mask = chunk['errors'].notna()
    if err_mask.any():
        for c, cnt in cid[err_mask].value_counts().items():
            client_error_count[c] += cnt
            
    # 7. Bill payments
    bill_mask = chunk['mcc'].isin(BILL_MCCS)
    if bill_mask.any():
        for c, cnt in cid[bill_mask].value_counts().items():
            client_bill_count[c] += cnt
            
    # 8. Date ranges & active months
    chunk['ym'] = chunk['date'].str.slice(0, 7)
    grp_dates = chunk.groupby('client_id')['date']
    for c, mn in grp_dates.min().items():
        if c not in client_min_date or mn < client_min_date[c]:
            client_min_date[c] = mn
    for c, mx in grp_dates.max().items():
        if c not in client_max_date or mx > client_max_date[c]:
            client_max_date[c] = mx
    for c, yms in chunk.groupby('client_id')['ym'].unique().items():
        client_year_months[c].update(yms)
        
    # 9. States
    st_mask = chunk['merchant_state'].notna()
    if st_mask.any():
        st_df = chunk.loc[st_mask, ['client_id', 'merchant_state']]
        st_counts = st_df.groupby(['client_id', 'merchant_state']).size()
        for (c, st), cnt in st_counts.items():
            client_states[c][st] += cnt

    print(f"Chunk {chunk_num} processed in {time.time() - t0:.2f}s (Total rows: {chunk_num * chunksize:,})")

print(f"Aggregating customer features for {len(client_txn_total)} clients...")

records = []
for cid in client_txn_total:
    txn_tot = client_txn_total[cid]
    online = client_online[cid]
    swipe = client_swipe[cid]
    chip = client_chip[cid]
    
    pos_cnt = client_pos_count[cid]
    pos_sum = client_pos_sum[cid]
    pos_amts = client_pos_amounts[cid]
    
    avg_amt = (pos_sum / pos_cnt) if pos_cnt > 0 else 0.0
    med_amt = float(np.median(pos_amts)) if len(pos_amts) > 0 else 0.0
    
    ref_cnt = client_refund_count[cid]
    ref_sum = client_refund_sum[cid]
    zero_cnt = client_zero_count[cid]
    
    err_cnt = client_error_count[cid]
    bill_cnt = client_bill_count[cid]
    
    act_months = len(client_year_months[cid])
    min_d = client_min_date[cid]
    max_d = client_max_date[cid]
    
    # Safe ratios
    dig_ratio = online / txn_tot if txn_tot > 0 else 0.0
    in_person_ratio = (swipe + chip) / txn_tot if txn_tot > 0 else 0.0
    refund_rate = ref_cnt / txn_tot if txn_tot > 0 else 0.0
    error_rate = err_cnt / txn_tot if txn_tot > 0 else 0.0
    bill_ratio = bill_cnt / txn_tot if txn_tot > 0 else 0.0
    txn_per_month = txn_tot / act_months if act_months > 0 else 0.0
    
    # Modal merchant state
    st_counter = client_states[cid]
    modal_state = st_counter.most_common(1)[0][0] if len(st_counter) > 0 else 'UNKNOWN'
    
    records.append({
        'client_id': cid,
        'Txn_Total': txn_tot,
        'Online_Count': online,
        'Swipe_Count': swipe,
        'Chip_Count': chip,
        'Digital_Ratio': dig_ratio,
        'InPerson_Ratio': in_person_ratio,
        'Total_Spend': pos_sum,
        'Avg_Amount': avg_amt,
        'Median_Amount': med_amt,
        'Refund_Count': ref_cnt,
        'Refund_Amount': ref_sum,
        'Refund_Rate': refund_rate,
        'Zero_Count': zero_cnt,
        'Error_Count': err_cnt,
        'Error_Rate': error_rate,
        'Bill_Count': bill_cnt,
        'Bill_Ratio': bill_ratio,
        'Active_Months': act_months,
        'Txn_per_Month': txn_per_month,
        'First_Txn_Date': min_d,
        'Last_Txn_Date': max_d,
        'Proxy_Region_State': modal_state
    })

df_txn_agg = pd.DataFrame(records)
df_txn_agg.to_csv('customer_transactions_aggregated.csv', index=False)
print(f"Aggregation saved to customer_transactions_aggregated.csv! Shape: {df_txn_agg.shape}")
print(f"Total time elapsed: {time.time() - start_time:.2f}s")
