import os, json
import glob

data_path = "remop_data/clean_berri_hn"

data_files = glob.glob(os.path.join(data_path, "*.json"))

single_attributes = dict()
compound_attributes = dict()
dataset_statistic = dict()

for data_file in data_files:
    dataset_name = os.path.basename(data_file)[:-len(".json")]
    print(dataset_name)

    data = [json.loads(d) for d in open(data_file)]
    data_length = len(data)

    print(data[0].keys())
    attributes = data[0]['attributes']
    for a in attributes:
        if a not in single_attributes:
            single_attributes[a] = data_length
        else:
            single_attributes[a] += data_length
    
    attributes_str = str(attributes)
    if attributes_str not in compound_attributes:
        compound_attributes[attributes_str] = data_length
    else:
        compound_attributes[attributes_str] += data_length
    
    dataset_statistic[dataset_name] = data_length

print(f"single_attributes: \n{json.dumps(single_attributes, indent=4)}")
print("="*8)
print(f"compound_attributes: \n{json.dumps(compound_attributes, indent=4)}")
print("="*8)
print(f"dataset_statistic: \n{json.dumps(dataset_statistic, indent=4)}")
    