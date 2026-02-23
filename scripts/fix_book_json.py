import json
import re

src = r"data/book.json"
out = r"data/book_fixed.json"

with open(src, 'r', encoding='utf-8') as f:
    text = f.read()

objects = []
idx = 0
n = len(text)

while True:
    # find next '{'
    start = text.find('{', idx)
    if start == -1:
        break
    brace = 0
    i = start
    in_string = False
    escape = False
    while i < n:
        ch = text[i]
        if ch == '"' and not escape:
            in_string = not in_string
        if not in_string:
            if ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    # include up to i
                    obj_text = text[start:i+1]
                    objects.append(obj_text)
                    idx = i+1
                    break
        if ch == '\\' and not escape:
            escape = True
        else:
            escape = False
        i += 1
    else:
        break

# Parse each object to ensure valid JSON and to normalize
parsed = []
for o in objects:
    try:
        parsed.append(json.loads(o))
    except Exception:
        # try to fix common issues: replace trailing commas inside objects/arrays
        cleaned = re.sub(r",\s*([}\]])", r"\1", o)
        try:
            parsed.append(json.loads(cleaned))
        except Exception as e:
            # skip invalid
            continue

# write output as single array
with open(out, 'w', encoding='utf-8') as f:
    json.dump(parsed, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(parsed)} objects to {out}")
