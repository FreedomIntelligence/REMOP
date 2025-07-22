import os, json

attributes = [
    "caption",
    "dialogue",
    "fact-verification",
    "legal",
    "medical",
    "news",
    "qa",
    "science",
    "sentence-paraphrase",
    "summarization",
    "wikipedia",
]

beir_mapping = {
    'nfcorpus': ["science", "qa"], 
    'msmarco': [], 
    'fiqa': ["qa"], 
    'scidocs': ["science", "summarization"], 
    'fever': [], 
    'arguana': ["fact-verification"], 
    'scifact': ["fact-verification", "science"], 
	'trec-covid': ["science", "qa"], 
	'climate-fever': ["fact-verification"], 
	'hotpotqa': [],  
	'nq': [], 
	'beir_mapping.py': [], 
	'quora': [], 
	'webis-touche2020': ["fact-verification"], 
	'cqadupstack': [], 
	'dbpedia-entity': ["wikipedia"]
}

if __name__ == '__main__':
    root_path = "beir_datasets"
    output_path = "beir_dpr_eval"
    for dataset, attributes in beir_mapping.items():
        dataset_path = os.path.join(root_path, dataset)
        if not os.path.isfile(os.path.join(dataset_path, "queries.jsonl")):
            continue
        queries = [json.loads(d) for d in open(os.path.join(dataset_path, "queries.jsonl"))]
        if not os.path.exists(os.path.join(output_path, dataset)):
            os.makedirs(os.path.join(output_path, dataset))
        with open(os.path.join(output_path, dataset, "queries.jsonl"), 'w') as writer:
            for query in queries:
                query['attributes'] = attributes
                writer.write(json.dumps(query, ensure_ascii=False))
                writer.write("\n")
        print(f"done {dataset} with length {len(queries)}")
        