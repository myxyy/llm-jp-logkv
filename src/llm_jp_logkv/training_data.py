"""Pinned SFT conversation conversion and uint32 caches on configured storage."""
import hashlib
from collections import Counter
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_ID = 'llm-jp/llm-jp-4.1-thinking-sft-data'
DATASET_REVISION = 'cb4210190af6a0000fd91cb5bb8361ad6d6ade01'
SUBSETS = ('llmjp_magpie_sft_v1.0', 'logical_math_coding_wizard8x22b',
           'llmjp_extraction_wiki_ja_v0.3')
SPLITS = ('reasoning_low', 'reasoning_medium', 'reasoning_high')
CONVERSATION_MODES = ('first', 'full')


def full_conversation(messages, *, reasoning_effort='low'):
    """Convert all complete turns; reject partial, tool, or nonalternating dialogue."""
    result, chunk, kwargs = [], [], None
    pending_user = False
    for message in messages:
        role, channel = message['role'], message.get('channel')
        if role in ('system', 'developer'):
            if result or pending_user:
                raise ValueError('System/developer messages must precede the conversation')
        elif role == 'user':
            if pending_user:
                raise ValueError('Multiple user messages before an answer')
            pending_user = True
        elif role == 'assistant' and channel in ('analysis', 'final', None):
            if not pending_user:
                raise ValueError('Assistant message without a pending user')
        else:
            raise ValueError('Unsupported message role/channel')
        chunk.append(message)
        if role == 'assistant' and channel in ('final', None):
            converted, turn_kwargs = first_exchange(chunk, reasoning_effort=reasoning_effort)
            result.extend(converted)
            if kwargs is None:
                kwargs = turn_kwargs
            chunk, pending_user = [], False
    if chunk or not result:
        raise ValueError('Incomplete conversation')
    return result, kwargs


def retained_analysis_template(original):
    """Adapt only the pinned template's past-turn branch; leave the model file intact."""
    marker = '{#- CoT is dropped during all previous turns, so we never render it for inference #}'
    if not isinstance(original, str) or original.count(marker) != 1:
        raise ValueError('Unrecognized chat template; cannot retain previous analysis safely')
    return original.replace(marker, '''{#- Retain previous analysis to match persistent chat state. #}
            {%- if "thinking" in message %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}''')


def first_exchange(messages, *, reasoning_effort='low'):
    """Convert public Harmony records to the original tokenizer template API.

    The pilot uses the first completed user/assistant exchange. Analysis becomes
    the template's thinking field; final content stays separate. Tool or unknown
    modalities are rejected rather than silently flattened into text.
    """
    kwargs = {'reasoning_effort': reasoning_effort, 'conversation_start_date': '2026-11-02'}
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


def prepare_cache(settings, output, *, max_length=2049, candidates_per_subset=100000, seed=20261102,
                  splits=SPLITS, conversation_mode='first'):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from .inference import file_hash, write_json

    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or any(split not in SPLITS for split in splits):
        raise ValueError('Choose distinct supported reasoning splits')
    if candidates_per_subset < 1 or max_length < 64:
        raise ValueError('candidates_per_subset must be positive and max_length must be >=64')
    if conversation_mode not in CONVERSATION_MODES:
        raise ValueError('Unknown conversation mode')
    root = settings.paths['DATASET_DIR'] / 'llm-jp-4.1-thinking-sft-data' / DATASET_REVISION
    paths = {}
    for subset in SUBSETS:
        for split in splits:
            paths[subset,split] = sorted((root/subset).glob(f'{split}-*.parquet'))
            if not paths[subset,split]:
                raise FileNotFoundError(f'No downloaded shards for {subset}/{split}')
    tokenizer = AutoTokenizer.from_pretrained(settings.model_path, local_files_only=True)
    template_kwargs = {}
    if conversation_mode == 'full':
        template_kwargs['chat_template'] = retained_analysis_template(tokenizer.chat_template)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if template_kwargs:
        (output/'chat_template.jinja').write_text(template_kwargs['chat_template'], encoding='utf-8')
    convert = full_conversation if conversation_mode == 'full' else first_exchange
    offsets, records, train, valid = [0], [], [], []
    counts, sources, seen = {}, [], set()
    rng = random.Random(seed)
    with (output/'tokens.bin').open('xb') as tokens_file:
        for subset in SUBSETS:
            reservoir, n = [], 0
            by_split = {split:{'source_rows':0,'candidates':0,'accepted':0,
                              'unsupported':0,'too_short_or_long':0,'duplicate':0} for split in splits}
            # One reservoir per subset across all selected splits, not N per split.
            for split in splits:
                for path in paths[subset,split]:
                    sources.append({'subset':subset,'split':split,'path':str(path),'sha256':file_hash(path)})
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                        for row in batch.to_pylist():
                            n += 1
                            by_split[split]['source_rows'] += 1
                            if len(reservoir) < candidates_per_subset:
                                reservoir.append((split,row))
                            else:
                                slot = rng.randrange(n)
                                if slot < candidates_per_subset:
                                    reservoir[slot] = (split,row)
                if not by_split[split]['source_rows']:
                    raise ValueError(f'No downloaded rows for {subset}/{split}')
            for split,row in reservoir:
                by_split[split]['candidates'] += 1
            counts[subset] = {'source_rows':n, 'candidates':len(reservoir), 'accepted':0,
                              'unsupported':0, 'too_short_or_long':0, 'duplicate':0,'splits':by_split}
            for begin in range(0,len(reservoir),128):
                texts, identifiers, groups, origins, turns = [], [], [], [], []
                for split,row in reservoir[begin:begin+128]:
                    try:
                        messages, kwargs = convert(json.loads(row['messages']),
                                                         reasoning_effort=split.removeprefix('reasoning_'))
                        # Group split by conversation content, independent of date/ID.
                        group = hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                             add_generation_prompt=False, **template_kwargs, **kwargs)
                    except (ValueError,KeyError,TypeError):
                        counts[subset]['unsupported'] += 1
                        by_split[split]['unsupported'] += 1
                        continue
                    texts.append(text); identifiers.append(row['ID']); groups.append(group)
                    origins.append(split)
                    turns.append(sum(m['role']=='user' for m in messages))
                if not texts:
                    continue
                encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)['input_ids']
                for ids, identifier, group, split, turn_count in zip(encoded, identifiers, groups, origins, turns):
                    if not 64 <= len(ids) <= max_length:
                        counts[subset]['too_short_or_long'] += 1
                        by_split[split]['too_short_or_long'] += 1
                        continue
                    if group in seen:
                        counts[subset]['duplicate'] += 1
                        by_split[split]['duplicate'] += 1
                        continue
                    seen.add(group)
                    index = len(records)
                    (valid if int(group[:8],16)%100 == 0 else train).append(index)
                    tokens = np.asarray(ids, dtype=np.uint32)
                    tokens_file.write(tokens.tobytes())
                    offsets.append(offsets[-1]+len(ids))
                    records.append({'subset':subset,'split':split,'id':identifier,'group_hash':group,
                                    'length':len(ids),'turns':turn_count})
                    counts[subset]['accepted'] += 1
                    by_split[split]['accepted'] += 1
            print(subset,counts[subset],flush=True)
    np.save(output/'offsets.npy',np.asarray(offsets,dtype=np.uint64))
    np.save(output/'train.npy',np.asarray(train,dtype=np.int64))
    np.save(output/'valid.npy',np.asarray(valid,dtype=np.int64))
    with (output/'records.jsonl').open('x',encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    generated_files = ['tokens.bin','offsets.npy','train.npy','valid.npy','records.jsonl']
    if template_kwargs:
        generated_files.append('chat_template.jinja')
    manifest = {'dataset_id':DATASET_ID,'revision':DATASET_REVISION,'splits':list(splits),
                'subsets':list(SUBSETS),'sources':sources,'counts':counts,'seed':seed,
                'candidates_per_subset':candidates_per_subset,
                'sampling':'uniform reservoir per subset over all selected splits; no fixed split quotas',
                'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                'source_files_sha256':{str(p):file_hash(p) for p in
                    (Path(__file__), Path('scripts/prepare_distillation_data.py'))},
                'train_examples':len(train),'valid_examples':len(valid),'tokens':offsets[-1],
                'max_length':max_length,'conversation_mode':conversation_mode,
                'past_analysis':'retained' if conversation_mode=='full' else 'not applicable (first exchange)',
                'turn_counts':{name:dict(Counter(str(records[i]['turns']) for i in indices))
                               for name,indices in (('all',range(len(records))),('train',train),('valid',valid))},
                'format':('uint32, variable length, complete conversation with all analysis, no truncation'
                          if conversation_mode=='full' else 'uint32, variable length, first completed exchange, no truncation'),
                'tokenizer_revision':settings.model_revision,
                'tokenizer_sha256':file_hash(settings.model_path/'tokenizer.json'),
                'template_sha256':file_hash(settings.model_path/'chat_template.jinja'),
                'effective_template_sha256':file_hash(output/'chat_template.jinja' if template_kwargs
                                                      else settings.model_path/'chat_template.jinja'),
                'files':{name:file_hash(output/name) for name in generated_files}}
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
