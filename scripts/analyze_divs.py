from bs4 import BeautifulSoup
import re, warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

with open(r"D:\Code\PROJECTS\FaithfulVoice\data\raw\NVDA_2024_10K.htm", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")

# List ALL div ids
divs_with_id = soup.find_all("div", attrs={"id": True})
print(f"ALL {len(divs_with_id)} divs with id=:")
for d in divs_with_id:
    id_val = d.get("id", "")
    text_preview = d.get_text(strip=True)[:120].replace("\n", " ")
    print(f'  id="{id_val}"  => "{text_preview}"')
