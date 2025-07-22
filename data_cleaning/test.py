import os, datasets, json

data_path = "remop_data/clean_berri_hn_jsonl"

dataset_list = os.listdir(data_path)

# d = datasets.load_dataset('json', data_files=[os.path.join(data_path, dataset) for dataset in dataset_list])
# ds = [datasets.load_dataset('json', data_files=[os.path.join(data_path, dataset)])['train'] for dataset in dataset_list]

# for dataset, d in zip(dataset_list, ds):
#     print(dataset, end=": ")
#     print([(type(d[f][0]), type(d[f][0][0])) for f in d.features])
#     # break

# print("done")


# for dataset in dataset_list:
#     data = [json.loads(d) for d in open(os.path.join(data_path, dataset))]
#     with open(f"remop_data/train_phase1_all/{dataset}", 'w') as writer:
#         for d in data:
#             if len(d['neg_ctxs']) < 1 or len(d['pos_ctxs']) < 1:
#                 continue
#             d['attributes'] = []
#             writer.write(json.dumps(d, ensure_ascii=False))
#             writer.write("\n")
#     print(f"done {dataset}")

for dataset in dataset_list:
    data = [json.loads(d) for d in open(os.path.join(data_path, dataset))]
    with open(f"remop_data/train_phase2_attr/{dataset}", 'w') as writer:
        for d in data:
            if len(d['neg_ctxs']) < 1 or len(d['pos_ctxs']) < 1:
                print(f"remove wrong data from {dataset}")
                continue
            # d['attributes'] = []
            writer.write(json.dumps(d, ensure_ascii=False))
            writer.write("\n")
    print(f"done {dataset}")