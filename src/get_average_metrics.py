import os
import json

import pandas as pd

folder = "/Users/odunayoogundepo/Desktop/jubilant-sniffle/runs/mafand_fewshot/results"
all_results = []


for file in os.listdir(folder):
    file_summary = {"bleu": [], "ter": [], "chrf": []}
    if file.endswith(".json"):
        with open(os.path.join(folder, file)) as f:
            data = json.load(f)
        
        for k, v in data.items():
            for sample in v:
                file_summary["bleu"].append(sample['metrics']["bleu"]['bleu'])
                file_summary["ter"].append(sample['metrics']["ter"]['score'])
                file_summary["chrf"].append(sample['metrics']["chrf"]['score'])
            
            for metric, list in file_summary.items():
                file_summary[metric] = sum(list) / len(list)
            
            file_summary['prompt_type'] = k.split('.txt_')[0]
            file_summary['translation_direction'] = k.split('.txt_')[1]
            all_results.append(file_summary)


        
file_name = "llama70b-mafand_all_results.csv"
pd.DataFrame(all_results).to_csv(file_name, index=False)

