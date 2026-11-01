# llm-jp実重みのLogKV初期推論（蒸留なし）

## 実験範囲と固定条件

`llm-jp/llm-jp-4.1-8b-thinking` の固定revision
`7c19bc90e49b717381e4f474ee95964c05217a28` をBF16で使用する。
全32層のAttentionを置換し、embedding、Q/K/V/O、元RoPE、Llama RMSNorm、FFN、
出力head、tokenizer、Harmony chat templateを流用する。蒸留・追加学習は行わない。

比較条件は、生成結果を見る前に次の3つに固定した。

1. **original**: 元Llamaのeager AttentionとHF generate。
2. **logkv**: 全32層LogKV、C=4、self slotあり、固定レベル減衰。
3. **logkv_no_bias**: 全32層LogKV、C=4、self slotあり、レベル減衰なし。

圧縮前に元RoPEを適用し、末尾QによるAttention poolingでK/Vをまとめる。
追加conv/gate/normは導入しない。元の8 KVヘッドを32 Qヘッドへ展開するため、
圧縮状態は32ヘッド分であり、8ヘッド共有キャッシュではない。
`UPSTREAM.json` の参照Attentionコアは無改変。

GPUはRTX 3090を1台だけ使用する。モデルは単一device上に配置し、量子化しない。
LogKVの生成は独自の明示的stateによるgreedy生成で、HF Cacheには未接続。

## 検証と品質指標

- CPU回帰は191件成功。独立FP64参照、streaming等価性、全モデルのdense等価性、
  重み同一性、EOS停止、NLLチャンク境界の計数を含む。
- GPU小モデルのBF16では、一括と7トークン分割の出力差max/meanがdense/LogKVとも0。
- 実重みについては、最初のチャット入力で全モデルのdense logitsを元Llamaと比較する。
  許容誤差はatol=0.125、rtol=0.02。観測誤差とargmax一致率も保存する。
- 生成は `evaluation/initial-quality.json` の日本語QA・算数・要約・短い文脈参照の4問。
  greedy、reasoning_effort=low、最大512新規トークン、EOS=[2,13]。
  chat template内の日付は再現用に2026-11-01へ固定する。
- LogKVのprefillは最大128トークンずつ。入力token IDs、出力token IDs、特殊トークン付き
  出力を保存する。analysisのみで上限に達したものは完了回答と扱わない。
- NLL/PPLは手書きの日本語説明文・算数説明文・英語説明文の3文章。
  元tokenizerの通常文章処理をそのまま使う。このtokenizerはBOSを自動追加しないため、
  各文章の先頭トークン以外を採点する。チャンク境界で追加の採点漏れはない。
  全採点トークンで加重平均したNLLからPPLを計算する。

これは小規模な初期状態の観察であり、標準ベンチマークや長距離参照性能の評価ではない。
生成時間はprefill込みで、元モデルとLogKVでは生成ループも異なる。速度比較のための
統一ベンチマークではない。品質結果と実行時間・メモリは分けて読む。

## 実行記録

現在は重み取得中。取得完了後に固定revisionのチェックサム検証と実モデル評価を行う。
結果は `${EXPERIMENT_DIR}/initial-quality-*` の新規ディレクトリへ記録する。
この節は実際の結果確認後に更新する。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./scripts/uv.sh run --locked llm-jp-logkv evaluate-initial \
  --device cuda:0 --max-new-tokens 512
```
