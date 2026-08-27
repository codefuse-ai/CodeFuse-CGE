from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer
from tqdm.auto import tqdm


tokenizer = AutoTokenizer.from_pretrained('models/qwen3-0.6b')
max_seq_length = 1023

output_dir = 'data_tokenized_qwen'
os.makedirs(output_dir, exist_ok=True)


def process_sent(sentence):

    tokenizer_outputs = tokenizer(sentence, max_length=max_seq_length, truncation=True, add_special_tokens=False)

    return np.array(tokenizer_outputs.input_ids + [tokenizer.eos_token_id])


def process_sent_batch(s):
    return s.apply(process_sent)

def parallelize(data, func, num_of_processes=8):
    indices = np.array_split(data.index, num_of_processes)
    data_split = [data.iloc[idx] for idx in indices]
    with Pool(num_of_processes) as pool:
        data = pd.concat(pool.map(func, data_split))
    return data


root_dir = 'datasets'

for ds_name in tqdm(sorted(parquet_files)):
    print(ds_name, flush=True)

    df = pd.read_parquet(f"{root_dir}/{ds_name}")
    df['query_input_ids'] = parallelize(df['query'], process_sent_batch, 62)

    num_neg = 24 if 'negative_2' in df.keys() else 1

    ls = df.passage.to_list()
    for i in range(1, num_neg+1):
        ls += df[f'negative_{i}'].to_list()
    ls = list(set(ls))
    df_tmp = pd.DataFrame({'text': ls})
    df_tmp['input_ids'] = parallelize(df_tmp['text'], process_sent_batch, 62)
    df_tmp = df_tmp.set_index('text')

    df['passage_input_ids'] = df.passage.map(df_tmp.input_ids)

    for i in range(1, num_neg+1):
        df[f'negative_{i}_input_ids'] = df[f'negative_{i}'].map(df_tmp.input_ids)

    df.to_parquet(f'{output_dir}/{ds_name}', index=False)
