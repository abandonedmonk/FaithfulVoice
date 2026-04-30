import requests
import json

headers = {"User-Agent": "FaithfulVoice/1.0 research@faithfulvoice.ai", "Accept-Encoding": "gzip, deflate"}

cik = "0001045810"
url = f"https://data.sec.gov/submissions/CIK{cik}.json"
r = requests.get(url, headers=headers, timeout=30)
data = r.json()

recent = data["filings"]["recent"]
print("=== ALL 10-K FILINGS ===")
for i, form in enumerate(recent["form"]):
    if form == "10-K":
        print(f"  accession={recent['accessionNumber'][i]}  "
              f"filingDate={recent['filingDate'][i]}  "
              f"reportDate={recent['reportDate'][i]}  "
              f"primaryDoc={recent['primaryDocument'][i]}")

print("\n=== ALL 10-Q FILINGS (recent) ===")
count = 0
for i, form in enumerate(recent["form"]):
    if form == "10-Q":
        print(f"  accession={recent['accessionNumber'][i]}  "
              f"filingDate={recent['filingDate'][i]}  "
              f"reportDate={recent['reportDate'][i]}  "
              f"primaryDoc={recent['primaryDocument'][i]}")
        count += 1
        if count >= 8:
            break
