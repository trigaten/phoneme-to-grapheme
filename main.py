
import pdfplumber
import re
pdf = pdfplumber.open('dictionary.pdf')
all_pages = [i.extract_text() for i in pdf.pages]

matches = []
for page in all_pages:
    matches += re.findall(r'[a-zA-Z0-9-] + /[\s\S]*/', str(page))


