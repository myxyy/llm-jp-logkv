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

**実重みでの全層LogKV推論と初期品質確認を完了した。**
推論は動作するが、今回の未蒸留C=4の両条件では有効な回答を生成できなかった。
減衰なしは固定減衰より短文NLLが良いものの、生成品質の回復には至っていない。

実行ソースcommit: `6f486873635044f922aeddc9254e065a64b3717d`（clean）。
Torch 2.11.0+cu128 / Transformers 5.4.0、RTX 3090 × 1、BF16。
モデル全重み4ファイルを含む必要12ファイルは `hf cache verify` でチェックサム一致。
remoteの未取得3ファイルは `.gitattributes` と補助Pythonスクリプト2本のみ。
実行では元のtokenizer.jsonとchat_template.jinjaを使用している。

成果物（ディレクトリ名の時刻はUTC、実行日はJSTで2026-11-02）:

- 評価: `/mnt/raid0/llm-jp-logkv/experiments/initial-quality-20261101T151841Z-010587d4/`
- 単発CLI: `/mnt/raid0/llm-jp-logkv/experiments/inference-20261101T152854Z-816a7dac/`
- ログ・チェックサム検証・最終CPU回帰:
  `/mnt/raid0/llm-jp-logkv/logs/initial-quality-20261101T145612Z-7449e41b/`

評価ディレクトリの `manifest.json` にsource/model revision、依存バージョン、
uv.lockとtokenizer/template等のハッシュ、設定、入力suiteを記録した。
`results.jsonl` は12生成と9文章NLLの全21記録、`summary.json` は損失集計、
`manual-review.json` は全12生成の目視判定、`dense-equivalence.json` は等価性検証。

実重みのdense等価性は93トークンでmax/mean logit差 **0**、argmax一致率 **100%**。
最終CPU回帰は **191 passed in 6.03s**。
単発 `infer` CLIは93+64 tokensを生成し、比較実験の固定減衰LogKVの先頭64トークンと
token IDが完全一致した。蒸留・最適化ステップ・重み更新は実施していない。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./scripts/uv.sh run --locked llm-jp-logkv evaluate-initial \
  --device cuda:0 --max-new-tokens 512
```

## 品質結果

| 条件 | 正しい最終回答（目視、4問） | EOS終了 | 平均NLL（182 tokens） | PPL |
|---|---:|---:|---:|---:|
| original | 4/4 | 4/4 | 2.4620 | 11.73 |
| logkv・固定減衰 | 0/4 | 0/4 | 4.8186 | 123.80 |
| logkv・減衰なし | 0/4 | 0/4 | 3.8005 | 44.73 |

| 質問 | original | 固定減衰LogKV | 減衰なしLogKV |
|---|---|---|---|
| 日本の首都 | 「東京です。」 | Answerとコロンを反復 | Answerなどの断片と0を反復 |
| 17−8＋5 | 「14個」 | 100000...と0を反復 | 別の質問を捏造した英語と0を反復 |
| 図書館の一文要約 | 日曜の時間と平日変更なしを正しく要約 | 空白・改行・タブを反復 | URL等の無関係な断片、0を反復 |
| 青チームの合言葉 | 「こはく」 | 改行と1を反復 | User、Answer、見出しを反復 |

LogKVの8生成はすべて512トークン上限に達し、Harmonyのfinalチャネルも生成しなかった。
生成logitsと評価損失は有限値で、NaN/Infによる停止やOOMは発生しなかった。
元の非Attention部分とdense経路の一致を確認した上での、未学習Attention置換の品質劣化である。
圧縮・RoPE・レベル減衰などの個別要因を完全に分離した実験ではない。
4問・182トークンの結果を、一般的な正答率や長文能力へ外挿しない。

## 実行時間とメモリ

| 条件 | 新規生成トークン/問 | prefill込み時間/問 | PyTorch peak allocated |
|---|---:|---:|---:|
| original | 19–97 | 0.70–3.20秒 | 16.031–16.044 GiB |
| logkv・固定減衰 | 512 | 65.17–66.08秒 | 16.030–16.034 GiB |
| logkv・減衰なし | 512 | 65.14–66.32秒 | 16.030–16.034 GiB |

全12生成の実行時間合計は約531秒。重みダウンロード・ロード・検証・NLL評価はこの値に含めない。
PyTorchの割当量はCUDA context等を含むGPU全使用量とは異なる。
LogKVの生成終了時の論理state tensor量は、4問順に6.00 / 7.50 / 5.25 / 7.50 MiB。
Q/K/Vと32 Qヘッド分を含み、両減衰条件で同じ。状態サイズは系列長のC進数の桁に依存する。
元モデルとは生成長が異なるため、これらの値から同一条件のキャッシュ削減率を主張しない。
現行のPython/PyTorch参照コアによる短文生成は高速化を達成しておらず、性能最適化は未実施。

## 単発推論の再現

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./scripts/uv.sh run --locked llm-jp-logkv infer \
  --prompt '日本の首都はどこですか。短く答えてください。' \
  --max-new-tokens 64 --conversation-start-date 2026-11-01
```

`--mode original` で元モデル、`--no-level-bias` で減衰なしLogKVを選べる。
現在の既定会話日付は初期実験再現用の2026-11-01。変更時は上のオプションで明示する。
各実行は新規のRAID成果物ディレクトリを作成し、既存結果を上書きしない。
今回の到達点は推論経路の実装と未蒸留状態の品質観察まで。蒸留・追加学習は行っていない。
