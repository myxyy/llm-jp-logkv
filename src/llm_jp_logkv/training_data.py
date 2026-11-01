"""Pinned SFT conversation conversion and uint32 caches on configured storage."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_ID = 'llm-jp/llm-jp-4.1-thinking-sft-data'
DATASET_REVISION = 'cb4210190af6a0000fd91cb5bb8361ad6d6ade01'
SUBSETS = ('llmjp_magpie_sft_v1.0', 'logical_math_coding_wizard8x22b',
           'llmjp_extraction_wiki_ja_v0.3')


def first_exchange(messages):
    """Convert public Harmony records to the original tokenizer template API.

    The pilot uses the first completed user/assistant exchange. Analysis becomes
    the template's thinking field; final content stays separate. Tool or unknown
    modalities are rejected rather than silently flattened into text.
    """
    kwargs = {'reasoning_effort': 'low', 'conversation_start_date': '2026-11-02'}
    result, analysis = [], None
    for message in messages:
        role, content = message['role'], message['content']
        if not isinstance(content, list):
            raise ValueError('Expected public SFT content list')
        text = []
        for part in content:
            if isinstance(part, str):
                text.append(part)
            elif part.get('type') == 'system_content' and role == 'system':
                for key in ('model_identity','knowledge_cutoff','conversation_start_date','reasoning_effort'):
                    if key in part:
                        kwargs[key] = part[key]
            elif part.get('type') == 'text':
                text.append(part['text'])
            else:
                raise ValueError('Unsupported content type')
        text = ''.join(text)
        if message.get('recipient') or message.get('tool_calls'):
            raise ValueError('Tool calling is outside this pilot dataset')
        if role in ('system','developer'):
            if text:
                result.append({'role':'developer','content':text})
        elif role == 'user':
            result.append({'role':'user','content':text})
        elif role == 'assistant' and message.get('channel') == 'analysis':
            if analysis is not None:
                raise ValueError('Multiple analysis messages before final')
            analysis = text
        elif role == 'assistant' and message.get('channel') in ('final',None):
            answer = {'role':'assistant','content':text}
            if analysis is not None:
                answer['thinking'] = analysis
            result.append(answer)
            if not any(m['role']=='user' for m in result):
                raise ValueError('No user prompt')
            return result, kwargs
        else:
            raise ValueError('Unsupported message role/channel')
    raise ValueError('No completed assistant response')


def prepare_cache(settings, output, *, max_length=2049, candidates_per_subset=30000, seed=20261102):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from .inference import file_hash, write_json

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = settings.paths['DATASET_DIR'] / 'llm-jp-4.1-thinking-sft-data' / DATASET_REVISION
    tokenizer = AutoTokenizer.from_pretrained(settings.model_path, local_files_only=True)
    offsets, records, train, valid = [0], [], [], []
    counts, sources, seen = {}, [], set()
    rng = random.Random(seed)
    with (output/'tokens.bin').open('xb') as tokens_file:
        for subset in SUBSETS:
            reservoir, n = [], 0
            for path in sorted((root/subset).glob('reasoning_low-*.parquet')):
                sources.append({'path':str(path),'sha256':file_hash(path)})
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        n += 1
                        if len(reservoir) < candidates_per_subset:
                            reservoir.append(row)
                        else:
                            slot = rng.randrange(n)
                            if slot < candidates_per_subset:
                                reservoir[slot] = row
            if not n:
                raise FileNotFoundError(f'No downloaded rows for {subset}')
            counts[subset] = {'source_rows':n, 'candidates':len(reservoir), 'accepted':0,
                              'unsupported':0, 'too_short_or_long':0, 'duplicate':0}
            for begin in range(0,len(reservoir),128):
                texts, identifiers, groups = [], [], []
                for row in reservoir[begin:begin+128]:
                    try:
                        messages, kwargs = first_exchange(json.loads(row['messages']))
                        # Group split by conversation content, independent of date/ID.
                        group = hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                             add_generation_prompt=False, **kwargs)
                    except (ValueError,KeyError,TypeError):
                        counts[subset]['unsupported'] += 1
                        continue
                    texts.append(text); identifiers.append(row['ID']); groups.append(group)
                if not texts:
                    continue
                encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)['input_ids']
                for ids, identifier, group in zip(encoded, identifiers, groups):
                    if not 64 <= len(ids) <= max_length:
                        counts[subset]['too_short_or_long'] += 1
                        continue
                    if group in seen:
                        counts[subset]['duplicate'] += 1
                        continue
                    seen.add(group)
                    index = len(records)
                    (valid if int(group[:8],16)%100 == 0 else train).append(index)
                    tokens = np.asarray(ids, dtype=np.uint32)
                    tokens_file.write(tokens.tobytes())
                    offsets.append(offsets[-1]+len(ids))
                    records.append({'subset':subset,'id':identifier,'group_hash':group,'length':len(ids)})
                    counts[subset]['accepted'] += 1
            print(subset,counts[subset],flush=True)
    np.save(output/'offsets.npy',np.asarray(offsets,dtype=np.uint64))
    np.save(output/'train.npy',np.asarray(train,dtype=np.int64))
    np.save(output/'valid.npy',np.asarray(valid,dtype=np.int64))
    with (output/'records.jsonl').open('x',encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest = {'dataset_id':DATASET_ID,'revision':DATASET_REVISION,'split':'reasoning_low',
                'subsets':list(SUBSETS),'sources':sources,'counts':counts,'seed':seed,
                'train_examples':len(train),'valid_examples':len(valid),'tokens':offsets[-1],
                'max_length':max_length,'format':'uint32, variable length, first completed exchange, no truncation',
                'tokenizer_revision':settings.model_revision,
                'tokenizer_sha256':file_hash(settings.model_path/'tokenizer.json'),
                'template_sha256':file_hash(settings.model_path/'chat_template.jinja'),
                'files':{name:file_hash(output/name) for name in ('tokens.bin','offsets.npy','train.npy','valid.npy','records.jsonl')}}
    write_json(output/'manifest.json',manifest)
    return manifest


class TokenCache(Dataset):
    def __init__(self, root, split, sequence_length):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.tokens = np.memmap(self.root/'tokens.bin',dtype=np.uint32,mode='r')
        self.offsets = np.load(self.root/'offsets.npy',mmap_mode='r')
        indices = np.load(self.root/f'{split}.npy')
        lengths = self.offsets[indices+1]-self.offsets[indices]
        self.indices = indices[(lengths>=2)&(lengths<=sequence_length+1)]
        if not len(self.indices):
            raise ValueError('No complete conversations fit the selected sequence length')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        row = self.indices[index]
        start,end = int(self.offsets[row]),int(self.offsets[row+1])
        return torch.from_numpy(self.tokens[start:end].astype(np.int64))


def collate_tokens(rows, *, sequence_length, pad_token_id):
    ids = torch.full((len(rows),sequence_length+1),pad_token_id,dtype=torch.long)
    lengths = torch.tensor([len(row) for row in rows],dtype=torch.long)
    for i,row in enumerate(rows):
        ids[i,:len(row)] = row
    return ids,lengths
