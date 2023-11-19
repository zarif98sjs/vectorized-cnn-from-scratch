## open json file
import json
with open('submission/1705010_online/temp.json', 'r') as f:
    data = json.load(f)
    print(data)