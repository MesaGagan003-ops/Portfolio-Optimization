import yfinance as yf
from collections import OrderedDict
import pandas as pd

nifty_tickers = [
    'RELIANCE.NS','TCS.NS','INFY.NS','HDFCBANK.NS','HDFC.NS',
    'ICICIBANK.NS','KOTAKBANK.NS','ITC.NS','HINDUNILVR.NS','BHARTIARTL.NS',
    'LT.NS','AXISBANK.NS','SBIN.NS','BAJFINANCE.NS','MARUTI.NS',
    'M&M.NS','SUNPHARMA.NS','TITAN.NS','TATASTEEL.NS','TATAMOTORS.NS',
    'WIPRO.NS','HCLTECH.NS','TECHM.NS','DIVISLAB.NS','DRREDDY.NS',
    'BRITANNIA.NS','EICHERMOT.NS','ULTRACEMCO.NS','POWERGRID.NS','NTPC.NS',
    'ONGC.NS','BPCL.NS','IOC.NS','JSWSTEEL.NS','JSWENERGY.NS',
    'GRASIM.NS','COALINDIA.NS','HINDALCO.NS','UPL.NS','ADANIENT.NS',
    'ADANIPORTS.NS','CIPLA.NS','BAJAJFINSV.NS','HDFCLIFE.NS','INDUSINDBK.NS',
    'CROMPTON.NS','GAIL.NS','SHREECEM.NS','SBILIFE.NS'
]

print(f'Total tickers to attempt: {len(nifty_tickers)}\n')

# Try downloading with progress
results = OrderedDict()
failed = []

for t in nifty_tickers:
    try:
        raw_t = yf.download(t, start='2022-01-01', progress=False)
        if raw_t is None or getattr(raw_t, 'empty', False):
            failed.append(t)
            print(f'✗ {t} — empty data')
        else:
            results[t] = raw_t
            print(f'✓ {t}')
    except Exception as e:
        failed.append(t)
        print(f'✗ {t} — {type(e).__name__}')

print(f'\n' + '='*70)
print(f'SUMMARY:')
print(f'='*70)
print(f'Successfully downloaded: {len(results)}')
print(f'Failed: {len(failed)}')

if failed:
    print(f'\nFailed tickers ({len(failed)}):')
    for idx, t in enumerate(failed, 1):
        print(f'  {idx}. {t}')
else:
    print('\nAll tickers downloaded successfully!')
