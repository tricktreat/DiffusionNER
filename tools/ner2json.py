import json

file_type = ["train", "dev", "test"]
wf_type = open(f"conll03_types.json","w")
types={"entities":{},"relations":{}}

for ty in  file_type:
    rf = open(f"{ty}.txt")
    wf = open(f"conll03_{ty}.json","w")
    
    datasets = []
    sample = {"tokens": [], "entities": [], "relations": []}
    idx = 0
    doc_id = 0
    start = end = None
    entity_type = None

    for line in rf:
        line = line.strip()
        if line == "-DOCSTART- -X- -X- O": # sep string
            doc_id += 1
            continue
        if line:
            last = idx
            fields = list(filter(lambda x:x,line.split(" ")))
            sample["tokens"].append(fields[0])
            sample["orig_id"] = str(doc_id)
            if fields[3].startswith("B-"):
                if start!=None and end!=None and end == idx:
                    sample["entities"].append({"start":start,"end":end,"type":entity_type})
                start = idx
                end = idx + 1
                entity_type = fields[3][2:]
                if entity_type not in types["entities"]:
                    types["entities"][entity_type]={"verbose": entity_type,"short": entity_type}
            if fields[3].startswith("I-"):
                end = end + 1
            if fields[3]=="O" and start!=None:
                sample["entities"].append({"start":start,"end":end,"type":entity_type})
                start = end = entity_type = None
            idx += 1
        else:
            if start!=None:
                sample["entities"].append({"start":start,"end":end,"type":entity_type})
                
            idx = 0
            start = end = None
            entity_type = None
            if len(sample["tokens"]):
                datasets.append(sample)
            
            sample = {"tokens": [], "entities": [], "relations": []}
    if len(sample["tokens"]):
        datasets.append(sample)    
    print(len(datasets))
    json.dump(datasets,wf)

print(len(types["entities"].keys()))
json.dump(types,wf_type)
