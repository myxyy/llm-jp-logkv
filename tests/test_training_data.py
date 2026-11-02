import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

from llm_jp_logkv import training_data as data


def messages(prompt):
    return [{'role':'user','content':[prompt]},
            {'role':'assistant','channel':'analysis','content':['reason']},
            {'role':'assistant','channel':'final','content':['answer']}]


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps({'prompt':messages[0]['content'],'effort':kwargs['reasoning_effort']})

    def __call__(self, texts, **kwargs):
        # Encode the effort as a large token ID, so split attribution is checked
        # through the written cache as well as the bookkeeping.
        ids=[]
        for text in texts:
            obj=json.loads(text)
            length=int(obj['prompt'].rsplit(':',1)[1])
            token={'low':196605,'medium':196606,'high':196607}[obj['effort']]
            ids.append([token]*length)
        return {'input_ids':ids}


@pytest.fixture
def source(tmp_path, monkeypatch):
    settings=SimpleNamespace(paths={'DATASET_DIR':tmp_path/'datasets'},model_path=tmp_path/'model',
                             model_revision='a'*40)
    settings.model_path.mkdir()
    for name in ('tokenizer.json','chat_template.jinja'):
        (settings.model_path/name).write_text('test tokenizer')
    root=settings.paths['DATASET_DIR']/'llm-jp-4.1-thinking-sft-data'/data.DATASET_REVISION
    for subset in data.SUBSETS:
        (root/subset).mkdir(parents=True)
        for split in data.SPLITS:
            rows=[{'ID':f'{subset}-{split}-{i}',
                   'messages':json.dumps(messages(f'{subset}/{split}/{i}:{length}'))}
                  for i,length in enumerate((63,64,100,2049,2050))]
            rows.append({'ID':'duplicate','messages':json.dumps(messages(f'{subset}/shared:100'))})
            rows.append({'ID':'unsupported','messages':json.dumps([{'role':'tool','content':['tool output']}])})
            pq.write_table(pa.Table.from_pylist(rows),root/subset/f'{split}-00000-of-00001.parquet')
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *args,**kwargs:Tokenizer())
    return settings,root


def test_all_splits_preserve_filters_metadata_and_effort(source, tmp_path):
    settings,_=source
    output=tmp_path/'cache'
    manifest=data.prepare_cache(settings,output)
    assert manifest['candidates_per_subset']==100000
    assert manifest['splits']==list(data.SPLITS)
    assert len(manifest['sources'])==9
    records=[json.loads(line) for line in (output/'records.jsonl').read_text().splitlines()]
    assert len(records)==30
    assert len({r['group_hash'] for r in records})==30
    for subset in data.SUBSETS:
        counts=manifest['counts'][subset]
        assert counts['source_rows']==counts['candidates']==21
        assert (counts['accepted'],counts['unsupported'],counts['too_short_or_long'],counts['duplicate'])==(10,3,6,2)
        for key in ('source_rows','candidates','accepted','unsupported','too_short_or_long','duplicate'):
            assert sum(c[key] for c in counts['splits'].values())==counts[key]
    tokens=np.memmap(output/'tokens.bin',dtype=np.uint32,mode='r')
    offsets=np.load(output/'offsets.npy')
    train=np.load(output/'train.npy')
    valid=np.load(output/'valid.npy')
    assert sorted([*train,*valid])==list(range(len(records)))
    assert not set(train)&set(valid)
    for i,row in enumerate(records):
        expected=196605+data.SPLITS.index(row['split'])
        assert tokens[offsets[i]]==expected
        assert offsets[i+1]-offsets[i]==row['length']
        assert (i in valid)==(int(row['group_hash'][:8],16)%100==0)
    for filename in ('train.npy','valid.npy'):
        indices=np.load(output/filename)
        if len(indices) and any(records[int(i)]['length']<=1025 for i in indices):
            selected=data.TokenCache(output,filename.removesuffix('.npy'),1024)
            assert len(selected)==sum(records[int(i)]['length']<=1025 for i in indices)


def test_reservoir_limit_is_per_subset_and_deterministic_across_splits(source, tmp_path):
    settings,_=source
    a=data.prepare_cache(settings,tmp_path/'a',candidates_per_subset=12)
    b=data.prepare_cache(settings,tmp_path/'b',candidates_per_subset=12)
    assert a['files']==b['files']
    for counts in a['counts'].values():
        assert counts['candidates']==12  # Not 12 per split.
        assert sum(c['candidates'] for c in counts['splits'].values())==12
        assert all(c['candidates']>0 for c in counts['splits'].values())


def test_low_only_reproduction_and_missing_split_rejection(source, tmp_path):
    settings,root=source
    manifest=data.prepare_cache(settings,tmp_path/'low',splits=('reasoning_low',),candidates_per_subset=30000)
    assert manifest['splits']==['reasoning_low']
    assert len(manifest['sources'])==3
    (root/data.SUBSETS[-1]/'reasoning_high-00000-of-00001.parquet').unlink()
    with pytest.raises(FileNotFoundError,match='reasoning_high'):
        data.prepare_cache(settings,tmp_path/'missing')
    assert not (tmp_path/'missing').exists()


@pytest.mark.parametrize('kwargs', [{'splits':()}, {'splits':('unknown',)},
    {'splits':('reasoning_low','reasoning_low')}, {'candidates_per_subset':0}, {'max_length':63}])
def test_invalid_cache_settings_rejected_before_writing(source, tmp_path, kwargs):
    settings,_=source
    with pytest.raises(ValueError):
        data.prepare_cache(settings,tmp_path/'bad',**kwargs)
    assert not (tmp_path/'bad').exists()


def test_explicit_system_effort_overrides_split_fallback():
    raw=[{'role':'system','content':[{'type':'system_content','reasoning_effort':'High'}]},*messages('prompt:64')]
    _,kwargs=data.first_exchange(raw,reasoning_effort='medium')
    assert kwargs['reasoning_effort']=='High'
