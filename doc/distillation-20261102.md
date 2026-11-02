# 固定減衰LogKVの蒸留（2026-11-02）

未蒸留の全層LogKVは初期4問で反復出力になったため、元モデルを教師とする
出力蒸留を実装した。今回のユーザー指示は「固定減衰・8時間・全6 GPU」であり、
通常の継続実験に対する最大2 GPUの制約は、この試行に限って明示的に変更された。
8時間の試行は3,126 stepsで正常終了した。全検証集合のKLは4.2622から0.4784に低下し、
初期比較と同条件の手書き4問では未蒸留0/4から蒸留後2/4となった。
元モデルは4/4であり、推論品質の保持・十分な回復を示す結果ではない。

## 実装と条件

| 項目 | 設定 |
|---|---|
| 教師 | llm-jp/llm-jp-4.1-8b-thinking、BF16、SDPA、全重み固定 |
| モデルrevision | `7c19bc90e49b717381e4f474ee95964c05217a28` |
| 生徒 | 全32層LogKV、C=4、元の固定レベル減衰、32 Qヘッド状態 |
| 学習対象 | Q/K/V/OのLoRA、rank=32、alpha=64、27,262,976 parameters |
| 維持する部分 | 元embedding、RoPE、RMSNorm、FFN、LM head、tokenizer、chat template |
| 損失 | 全196,608語彙のKL(teacher \|\| student)、T=1、+ 0.1 × next-token CE |
| 対象トークン | paddingを除く全next-token位置。prompt/analysis/finalを含む |
| 精度・メモリ | BF16演算、LoRA master FP32、decoder層と語彙損失の非reentrant checkpoint |
| 並列化 | 6 GPU DDP、GPUあたり1会話、4回勾配蓄積、global batch=24 |
| 系列長 | 入力1,024、次トークンラベル用を含め1,025 tokensまでの完結会話 |
| 損失分割 | 512 positionsごと。語彙のtop-k近似は使わない |
| optimizer | fused AdamW、lr=1e-4、betas=(0.9,0.95)、weight decay=0.01、clip norm=1 |
| schedule | 50 steps warmup、経過wall timeによるcosine decay、最小lr比0.1 |
| 記録 | 200 optimizer stepsごとにcheckpoint・固定検証集合・6問のgreedy生成 |
| 生成上限 | 定期サンプル256 tokens、終了後の初期4問比較512 tokens |

教師と生徒は凍結した重みだけを共有する。LoRAを教師のself_attnへ取り付けず、
生徒の独立したAttention adapterの射影だけに追加する。
学習時は状態を保持しないgrouped LogKV、生成時は既存streaming APIを使う。
`logkv/attention.py` は `UPSTREAM.json` のSHA-256と一致する無改変コアのまま。
DDPの平均化と勾配蓄積に合わせて、全rankの有効トークン数で損失を正規化する。

8時間は初期検証・サンプル、定期保存・検証・生成を含む訓練開始後のwall time。
起動時のモデル読込は含まず、終了時の保存・検証・生成と別プロセスの最終比較が
短い後処理として加わる。再開時も元の開始時刻を保持し、停止中の時間を含めるため、
再開のたびに8時間の予算を追加しない。

## データ

公開SFTデータ `llm-jp/llm-jp-4.1-thinking-sft-data` のrevision
`cb4210190af6a0000fd91cb5bb8361ad6d6ade01` を固定した。
`llmjp_magpie_sft_v1.0`、`logical_math_coding_wizard8x22b`、
`llmjp_extraction_wiki_ja_v0.3` の reasoning_low shardsから、各30,000件を
seed=20261102でreservoir samplingする。

最初の完結したuser/assistant応答を使い、analysisは元templateのthinking、
finalはcontentに分ける。元データのsystem metadataも保持する。
会話途中の切り捨て、tool会話、元プロジェクトのuint16キャッシュは使わない。
本文hashで重複除去し、同じhashに基づいて約1%を検証集合へ分離する。
生データshard、tokenizer、template、生成キャッシュのhashをmanifestに記録する。

キャッシュ全体はtrain=77,648、valid=818、54,435,569 tokens（uint32）。
今回の長さで完結する会話に絞るとtrain=62,364、valid=674。
定期検証は固定の48会話、開始・終了時は全674会話を検証する。
この選別は短い応答へ偏るため、長距離品質の評価には使えない。

実キャッシュの再確認では、全54,435,569 tokensのID範囲は2〜196,607でuint32を維持。
offsetの整合性、会話hashの一意性、学習・検証indexの非重複を確認した。
各subsetから1件ずつ抽出し、analysis/finalチャネルと終了returnトークンを確認した。
監査記録は本試行run内の `reviews/data-cache-audit.json` と
`reviews/data-cache-index-audit.json`。後者では全offset区間と記録された長さの一致、
全レコードが学習・検証のどちらかにちょうど1回含まれることも確認した。

保存先:

```text
/mnt/raid0/llm-jp-logkv/datasets/distill-cache-20261101T155341Z-4326ce39
```

再構築する場合は必ず新しい出力先を指定する。

```bash
HF_HUB_DISABLE_XET=1 ./scripts/uv.sh run --locked python scripts/prepare_distillation_data.py \
  --download --output /mnt/raid0/llm-jp-logkv/datasets/NEW-CACHE
```

## 実行と再開

現在の既定値は **時間制限なし・1 epoch**。`--max-hours` の省略は無制限、
`--num-epochs` の省略は1となる。`--num-epochs N` は再開前からの通算周回数を表す。
時間やstepの上限も指定した場合は、最初に達した条件で保存して終了する。
以下のコマンドは既定の手動起動とは別に、前回の8時間試行の条件を明示したもの。

既存成果物を上書きしないため、run-dirとlog-dirは新しい名前にする。
これは今回承認された6 GPUのコマンドであり、将来の長時間実験の包括承認ではない。

```bash
./scripts/uv.sh run --locked python scripts/run_distillation.py \
  --gpus 0,1,2,3,4,5 \
  --log-dir /mnt/raid0/llm-jp-logkv/logs/NEW-RUN \
  --evaluate-final -- \
  --data /mnt/raid0/llm-jp-logkv/datasets/distill-cache-20261101T155341Z-4326ce39 \
  --run-dir /mnt/raid0/llm-jp-logkv/experiments/NEW-RUN \
  --num-epochs 0 --max-hours 8 --sequence-length 1024 --batch-size 1 --grad-accum 4 \
  --rank 32 --alpha 64 --chunk-size 4 --loss-chunk-tokens 512 \
  --temperature 1 --ce-weight 0.1 --lr 1e-4 --warmup-steps 50 \
  --sample-every 200 --save-every 200 --validate-every 200 \
  --val-examples-per-rank 8 --sample-max-new-tokens 256 --seed 20261102
```

supervisorはtorchrunの終了コードを保存し、成功時には最終adapterと元モデルの
比較をGPU 0で行う。切断後も実行する場合はsupervisorを別sessionで起動する。
`process.json` にPIDとLinuxのprocess start ticksを記録する。
監視呼び出しのtimeoutだけで再起動せず、PIDの一致とmetricsの更新を確認する。

checkpointは `${CHECKPOINT_DIR}/RUN/step-NNNNNNN/` に保存する。
`adapter.pt`、optimizer/RNG/data cursor入り `training_state.pt`、設定、
チェックサム入り `complete.json` が揃ったものだけを使用する。
再開は同じ並列数・訓練設定と新しいrun/log-dirに `--resume CHECKPOINT` を指定する。
epoch数・時間・stepの上限は再開時に変更できる。旧checkpointに記録された `max_hours=8` は
訓練設定の互換性検査から分離したため、`--max-hours` を省いて時間上限を外せる。
モデル・データ・batch size・勾配蓄積数・optimizer/LR係数などの整合性検査は維持する。
`--num-epochs` は通算なので、既に1 epoch以上を完了したcheckpointから追加学習するには
2以上など、現在の進捗より大きい上限を指定する。
`--max-hours` を明示した場合だけ、停止時間を含めた元の開始時刻から上限を計算する。

学習率は有限のepoch上限がある場合はepoch進捗によるcosine decayを使う。
`--num-epochs 0` では明示された時間上限、時間上限がなければstep上限を使う。
時間・step上限の両方がない `--num-epochs 0` はエラーにする。
再開時に期間の設定を変えた場合は、新しい期間に合わせた学習率となる。
旧8時間試行のsourceは `ee47991` であり、この既定値・epoch境界処理の変更前の結果である。

epoch末尾は残りのミニバッチ数で最後の勾配蓄積・DDP同期を行い、次のepochへ食い込まない。
再開時は保存されたデータ位置から残りを処理する。DDP samplerとDataLoaderの `drop_last`
設定は従来どおりで、GPU台数・batch sizeで割り切れないデータ末尾は各epochの対象外となる。
起動ログと `started.json` に時間・epoch上限を、metricsにepoch・データ位置・実際の蓄積回数を記録する。

この既定値変更はCPU 233 testsで検証した。実重み2 GPU・短い10会話の検証では、
4＋1回の勾配蓄積で1,231有効tokensを重複なく1周し、`reason="num_epochs"` で終了した。
保存・中断から残りだけ再開した最終重みも連続実行と完全一致した。
旧版の期限切れcheckpointから時間制限を外す再開も確認済み。
記録は `logs/epoch-default-20261102T104125Z-validation.json`（RAID配下）。

### control.cmdによる保存・中断（試行完了後に追加）

手動操作は次のとおり。uv wrapperの実行ディレクトリであるプロジェクト直下を既定の場所とする。

```bash
cd /home/myxy/llm-jp-logkv
printf 'save_and_exit\n' > control.cmd
```

rank 0だけがコマンドを読み、全rankへ同じ終了理由をbroadcastする。
実行中のoptimizer step・検証・サンプル生成が終わったstep境界で、adapter、optimizer、
各rankのRNG、epoch内のデータ位置を保存して終了コード0で停止する。
受理したファイルは削除し、`completed.json` に `reason="save_and_exit"` とcheckpointパスを記録する。
この場合の終了後全件検証と生成は省き、`validation` はnullになる。
supervisorに `--evaluate-final` があっても比較評価を省き、`evaluation-skipped.json` を記録する。

起動コマンドの `--` より後に `--control-file PATH` を追加すると監視先を変更できる。
複数runは同じ制御ファイルを共有せず、たとえばrunごとの絶対パスを指定する。
空・未認識の内容では停止せず、ファイルも削除しない。対応コマンドは `save_and_exit`。
既存のrun-dir内 `STOP` による停止は引き続き使え、従来どおり終了時の検証・生成を実施する。
新規起動のコピー可能な2 GPUコマンドは[README](../README.md#固定減衰logkvの蒸留)に記載した。

追加機能の検証: CPU **214 passed in 10.91s**。2 rankのGlooでコマンドをrank 0だけが
読み、全rankへ同じ終了理由が伝わることを確認した。実重み2 GPUの短いsmokeでは
step 1でコマンドを発行して保存・正常終了し、別runからstep 2まで再開できた。
optimizerのstepとデータ位置が1→2へ進み、元の開始時刻と訓練設定を維持している。
中断時は終了後評価が省かれ、再開後の通常終了では検証・生成・比較評価が実行された。
生成上限4、系列長128、LoRA rank 2の操作検証であり、品質実験ではない。
保存・再開の両checkpointはchecksum一致、各runの2 rankの重みも一致した。
記録は `/mnt/raid0/llm-jp-logkv/logs/control-command-validation-20261102.json`。
run名は `control-smoke-20261102T014430Z` と `control-resume-20261102T014553Z` で、
同じRAIDルートの `experiments/`、`logs/`、`checkpoints/` に成果物を保存した。

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/uv.sh run --locked llm-jp-logkv infer \
  --adapter /mnt/raid0/llm-jp-logkv/checkpoints/RUN/step-NNNNNNN \
  --prompt '日本の首都はどこですか。短く答えてください。' --max-new-tokens 512
```

## 検証記録

- CPU: **200 tests passed**。FP64 grouped/streaming出力・勾配一致、teacher不変、
  LoRA更新後の全モデルgrouped/streaming一致（FP64とBF16）、
  checkpointing/paddingの損失・勾配、独立したKL/CE式（T=1,2）、
  196,607を含むuint32 token ID保持を検証した。
- 6 GPUのNCCL all-reduce成功。
- 実装commit: `aeb4911`。初回smokeは3 stepsで正常終了し、6 rankの最終adapter
  hashが一致した。step 2のcheckpointから別runへ再開し、step 3のKL/CE/
  gradient norm/有効トークン数が初回の同じstepと完全一致した。
  4 stepsまで更新後も6 rankが一致し、定期・終了時サンプル各6問の保存を確認。
  最終checkpointの `evaluate-initial --adapter` も終了コード0だった。
  smokeの生成上限は16 tokensで、品質評価には使用しない。
  成果物は `experiments/distill-smoke-20261102-01` と
  `experiments/distill-resume-smoke-20261102-01`、ログは同名の `logs/` 以下。
  いずれも `/mnt/raid0/llm-jp-logkv/` 配下。これらはコミット前の検証であり、
  各manifestにその時点のソースhashを記録している。
- 6 GPU・実重み・本設定での短時間訓練は1 step約7.8秒、
  約1,570〜1,740有効tokens/s、全GPU使用率100%、使用メモリ各約20.3 GiBを観測。
  paddingを含むtokens/sとは区別する。短時間の測定であり8時間平均ではない。
- 損失分割幅の測定は
  `/mnt/raid0/llm-jp-logkv/logs/distill-loss-chunk-benchmark-20261102.json`。
  同測定のlossは設定間で更新を継続したため、品質比較には使用しない。

## 8時間試行の実行記録

- Run: `distill-fixed-20261101T162321Z-5dfd0965`
- Source: `ee47991e1b3ed6d194bd704e44a3dde1be5dbbd2`、開始時clean。
- 訓練開始: **2026-11-02 01:23:50 JST**。
- 訓練deadline: **2026-11-02 09:23:50 JST**。最終更新は3,126 steps、経過28,801.08秒。
  step境界で `deadline` により終了し、再開・設定変更はなかった。
- 最終保存・全件検証・6問生成: **09:26:01 JST**完了、訓練終了コード0。
- 元モデルとの別プロセス比較: **09:26:50 JST**完了、評価終了コード0。
- 6 GPU稼働とteacher固定・LoRA 27,262,976 parametersをruntimeで確認。
- 初期の固定48会話: KL=4.2372867、CE=5.3847451、26,443 tokens。
  全674会話の開始・終了時比較は `validation.jsonl` の `*_all_heldout` を使用する。

| 内容 | 保存先 |
|---|---|
| manifest、step metrics、validation、生成全文 | `/mnt/raid0/llm-jp-logkv/experiments/distill-fixed-20261101T162321Z-5dfd0965/` |
| torchrunログ、PID、終了コード | `/mnt/raid0/llm-jp-logkv/logs/distill-fixed-20261101T162321Z-5dfd0965/` |
| checkpoint | `/mnt/raid0/llm-jp-logkv/checkpoints/distill-fixed-20261101T162321Z-5dfd0965/` |

`samples/initial-step-0000000.json`、`samples/periodic-step-NNNNNNN.json`、
`samples/final-step-NNNNNNN.json` に6問を保存する。訓練終了後はrunの
`completed.json` とログの `training-exit.json`、`evaluation-exit.json`、
`supervisor-completed.json` を確認する。最終比較の新規artifact directoryは
`final-evaluation.log` に出力される。起動済みrunを重複起動しない。

最終checkpointは `step-0003126`。200〜3,000 stepsの15回と合わせて16個を保存した。
サンプルは初期・15回の定期・最終の計17ファイル、各6問で計102生成を記録した。
6 rankの最終adapter tensor hashは一致し、最終adapterを別プロセスで読み込んで評価した。
最終adapterファイルのSHA-256は
`c80a580201292b1312daa1f03ce720db1051b2e793abe5f3b8fe7d268272a88d`。

### 定期記録の途中結果

約32分経過時点の最初の定期保存を確認した。学習・optimizer state両ファイルの
SHA-256は `complete.json` と一致し、訓練は200 steps以降も継続している。
固定48会話・26,443 tokensの検証値は以下のとおり。

| step | KL | CE |
|---|---:|---:|
| 0 | 4.237287 | 5.384745 |
| 200 | 0.862019 | 1.844732 |
| 400 | 0.707254 | 1.681957 |
| 600 | 0.645481 | 1.614846 |
| 800 | 0.593366 | 1.569258 |
| 1000 | 0.562368 | 1.530876 |
| 1200 | 0.536621 | 1.504088 |
| 1400 | 0.517838 | 1.491023 |
| 1600 | 0.502905 | 1.477193 |
| 1800 | 0.491404 | 1.463755 |
| 2000 | 0.483557 | 1.452208 |
| 2200 | 0.477677 | 1.450476 |
| 2400 | 0.471551 | 1.444897 |
| 2600 | 0.468816 | 1.440404 |
| 2800 | 0.464881 | 1.438244 |
| 3000 | 0.462769 | 1.434686 |

200 stepsでは、256 tokens上限の6問はすべてfinalチャネルを出しEOSで終了した。
一方、正しい最終回答は首都の1問だけで、元の4問では1/4、新規2問を含めると1/6。
反復・未完了の出力からは変化したが、内容の正確さはまだ不十分である。

| 質問 | 200 stepsの最終回答・判定 |
|---|---|
| 首都 | 「東京」: 正答 |
| 17−8＋5 | 「5個」: 誤答。思考文でも入力の個数を取り違えた |
| 図書館の要約 | 「日曜は閉館」: 提示文と逆の内容 |
| 青チームの合言葉 | 「青」: 正答の「こはく」を取り出せていない |
| 英文和訳 | 元の英語文をそのまま出力 |
| 9、2、15、4の整列 | 多数の無関係な数を出力 |

生の出力はrun内 `samples/periodic-step-0000200.json`、目視判定は
`reviews/step-0000200.json`、保存確認は `reviews/checkpoint-0000200.json`。
これは途中の手書きsmoke評価であり、一般的な正答率や推論能力の保持を示すものではない。
その後も同じ設定で8時間の試行を継続した。

400 stepsでもEOS終了は6/6、正しい最終回答は首都の1問だけ（元の4問では1/4）。
算数は17−8=9までは正しいが、入力にない手順を追加して16個と回答した。
要約・合言葉・和訳・整列にも誤りが残る。検証損失の低下を、回答品質の回復と同一視しない。
`samples/periodic-step-0000400.json` と `reviews/step-0000400.json` に記録した。
400 stepsのadapterとoptimizer stateも保存manifestのSHA-256に一致した。

600 stepsもEOS終了6/6、正しい最終回答1/6（元の4問では1/4）。
合言葉は緑チームの「すみれ」を青チームの回答にし、算数は思考文と最終回答の数値も
一貫していない。詳細は `reviews/step-0000600.json`。保存済み2ファイルのSHA-256一致も確認した。

800 stepsはEOS終了6/6、**最終回答の内容だけなら2/6**（元の4問では2/4）。
首都に加えて算数の最終数値が14個になったが、analysisでは渡す数を取り違えて17個とし、
finalと矛盾する。finalの説明でも「買う」を「友だちからもらう」に変えている。
要約は日曜開館と開始10時を保持したが、終了16時と平日変更なしを落とした。
合言葉・和訳・整列は誤答。数値の正答を推論の健全性と同一視しない。
詳細は `reviews/step-0000800.json`。800 stepsの保存済み2ファイルもSHA-256一致を確認した。

1,000 stepsもEOS終了6/6、最終回答の内容では2/6（元の4問では2/4）。
算数のfinalは入力条件と計算を正しく記述するようになったが、analysisは17個とし矛盾が残る。
要約は日曜10〜16時を保持する一方、提示文にない時間拡大を追加した。
詳細は `reviews/step-0001000.json`。保存済み2ファイルのSHA-256一致も確認した。

1,200 stepsはEOS終了6/6、正しい最終回答1/6（元の4問では1/4）に戻った。
算数は渡す数を5個とし、入力にない手順も加えて22個と回答した。
要約は平日の開館時間を延長すると誤って追加する。損失の改善と生成品質は単調に対応しない。
詳細は `reviews/step-0001200.json`。保存済み2ファイルのSHA-256一致も確認した。

1,400 stepsもEOS終了6/6、正しい最終回答1/6（元の4問では1/4）。
算数は22個、合言葉は提示文にない「はくちょう」と回答した。要約にも未提示の時間を追加する。
詳細は `reviews/step-0001400.json`。保存済み2ファイルのSHA-256一致も確認した。

1,600 stepsはEOS終了6/6、正しい最終回答2/6（元の4問では1/4）。
首都に加えて和訳が「図書館は月曜日に閉まっています。」と正しくなった。
算数は17個、合言葉は赤チームの「さくら」と回答し、文脈に沿った参照には課題が残る。
詳細は `reviews/step-0001600.json`。保存済み2ファイルのSHA-256一致も確認した。

1,800 stepsもEOS終了6/6、正しい最終回答は首都と和訳の2/6（元の4問では1/4）。
算数はanalysisで14個と正しく計算したが、finalでは入力と計算を取り違えて19個と回答した。
思考文の正しい計算も、最終回答の正確さを保証していない。
詳細は `reviews/step-0001800.json`。保存済み2ファイルのSHA-256一致も確認した。

2,000 stepsもEOS終了6/6、正しい最終回答は首都と和訳の2/6（元の4問では1/4）。
和訳には助詞の不自然さがあるが、月曜休館という内容は保持する。
算数はanalysisの14個とfinalの17個が不一致。整列は4,9,15となり、入力の2を落とした。
要約は日曜開館だけを述べ、時刻と平日変更なしを落としている。
詳細は `reviews/step-0002000.json`。保存済み2ファイルのSHA-256一致も確認した。

2,200 stepsもEOS終了6/6、正しい最終回答は首都と和訳の2/6（元の4問では1/4）。
要約は日曜開館と平日変更なしを保持したが、日曜10〜16時を落とした。
算数はanalysisとfinalの両方で渡す数を5個と取り違え、17個と回答した。
詳細は `reviews/step-0002200.json`。保存済み2ファイルのSHA-256一致も確認した。

2,400 stepsもEOS終了6/6、正しい最終回答は首都と和訳の2/6（元の4問では1/4）。
算数はanalysisで14個と計算してもfinalは17個となる。合言葉は緑チームの「すみれ」。
要約は引き続き日曜の時刻を落とし、整列は入力にない数を含む。
詳細は `reviews/step-0002400.json`。保存済み2ファイルのSHA-256一致も確認した。

2,600 stepsもEOS終了6/6、正しい最終回答は首都と和訳の2/6（元の4問では1/4）。
算数はanalysisの14個に対しfinalは12個。要約には未提示の土曜開館と10〜17時が混入した。
詳細は `reviews/step-0002600.json`。保存済み2ファイルのSHA-256一致も確認した。

2,800 stepsもEOS終了6/6、正しい最終回答は首都と和訳の2/6（元の4問では1/4）。
算数は17個。要約は平日に来館できないと誤り、平日と日曜の開館時間も未提示の内容に変えた。
詳細は `reviews/step-0002800.json`。保存済み2ファイルのSHA-256一致も確認した。

3,000 stepsもEOS終了6/6、正しい最終回答は首都と和訳の2/6（元の4問では1/4）。
算数はanalysisの14個とfinalの17個が不一致。要約は未提示の平日9〜17時を追加した。
詳細は `reviews/step-0003000.json`。保存済み2ファイルのSHA-256一致も確認した。

## 最終品質

開始・終了で同じ全674会話、347,922有効トークンを使った。
定期記録の48会話とは母集団が異なるため、下表同士で比較する。

| 全検証集合 | KL | next-token CE |
|---|---:|---:|
| 学習前 | 4.262197 | 5.442251 |
| 3,126 steps後 | 0.478394 | 1.470831 |

最終6問はEOS終了6/6、正しい最終回答2/6（首都と和訳）。算数はanalysisが14個、
finalが17個と矛盾し、合言葉は緑チームの「すみれ」を答えた。
`reviews/final-step-0003126.json` に全文に基づく判定を残した。

元の初期比較と同じ4問・生成上限512・chat date=2026-11-01で、最終adapterを再評価した。
元モデルの生成token IDsと文章NLLは初期比較と完全一致した。
モデルrevision・tokenizer・template・質問・入力token IDs・生成設定も同じである。

| 条件 | 正しい最終回答 | EOS終了 | 固定3文章のPPL（182 tokens） |
|---|---:|---:|---:|
| 元モデル | 4/4 | 4/4 | 11.7287 |
| 未蒸留・固定減衰LogKV | 0/4 | 0/4 | 123.7955 |
| 蒸留後・固定減衰LogKV | 2/4 | 4/4 | 54.4838 |

蒸留後の正答は首都と要約。要約は日曜10〜16時と平日変更なしを保持したが、
「来月」は省略した。算数の最終回答は17個、合言葉は「すみれ」で誤答。
最終6問の方はchat date=2026-11-02で、要約は16時を落としていた。
日付条件が異なる2系列の正答数は混ぜず、上表では初期比較と条件を揃えた結果を使う。
手書きの短い設問であり、一般的な正答率・長距離品質・推論能力の保持を示さない。

最終比較の成果物:

```text
/mnt/raid0/llm-jp-logkv/experiments/initial-quality-20261102T002607Z-cf63ef05/
```

同ディレクトリの `results.jsonl` に全文とtoken IDs、`manual-review.json` に8生成の判定、
`summary.json` にNLL/PPLを保存した。`dense-equivalence.json` は実重み93 tokensで
元Llamaとのlogits最大誤差0、argmax一致100%を再確認している。

## 実行性能と検証の範囲

- 3,126 optimizer steps、75,024会話の提示（62,364件に対し約1.203周分）、
  **38,285,241有効tokens**を処理した。paddingはこのtoken数に含めない。
- 定期評価・保存・生成を含む8時間で平均 **1,329.30有効tokens/s**。
  optimizer stepの時間だけでは平均 **1,357.02有効tokens/s**、1 step平均9.025秒。
- GPU計測157回で、使用率の中央値は6台とも100%、使用メモリ中央値は各20,801 MiB。
  PyTorchのrank最大割当ピークは19.439 GiB。使用率は標本値であり、全瞬間の保証ではない。
- 最終4問の生成時間は元モデル0.67〜3.42秒、蒸留後LogKV4.49〜14.40秒。
  出力長も異なるため、制御された速度比較や高速化の実証とは扱わない。
- 終了後のCPU回帰テストは **200 passed in 9.51s**。実重みBF16比較の追加確認に伴い、
  独立FP64参照・grouped/streaming・LoRA・勾配・データ型の既存テストを再確認した。
  ログとJUnit XMLはこのrunの `logs/` 以下の `cpu-tests-final.*`。
- 訓練開始時の全 `src/` hash、`uv.lock`、`UPSTREAM.json` の全移植ファイルの一致を確認した。
  実行中に変更したGit管理ファイルは実験記録のみで、訓練コードと参照コアは同一である。

### 実重みでの一括計算と分割計算

最終adapterを読み込み、学習時のgrouped経路と、128トークン分割＋末尾8トークンの
1トークンずつの推論経路を比較した。実際の首都93 tokens、算数110 tokens、
最長の検証会話1,023入力tokensを使い、状態のoffsetも確認した。

GPU BF16では事前に設定した `atol=0.125, rtol=0.02` に3入力とも未達だった。
logits最大差0.500〜1.016、平均差0.0356〜0.0425、argmax一致99.09〜100%、
平均KL(grouped || streaming)は0.000587〜0.001338だった。
この実重みでの厳しい誤差判定を「一致」とは扱わず、BF16の制限として残す。
CPU FP32では同じ入力と経路が、事前設定 `atol=rtol=1e-4` に3入力とも合格した。
最大差は93 tokensで3.41e-5、110 tokensで3.58e-5、1,023 tokensで1.31e-4
（相対許容差込みで合格）。平均差2.03e-6〜2.72e-6、argmax一致は全入力100%だった。
dtypeとdeviceの両方が異なる比較なので、差の原因をこれだけで一意に断定しない。
訓練と生成の数値経路を点検できた一方、GPU BF16での完全一致や一般的な誤差上限は保証しない。

検証スクリプトと各入力の数値・token IDsはrun内 `reviews/verify-real-streaming*.py`、
`reviews/real-streaming-comparison*.json`。これは数値経路の確認であり、生成品質の評価とは別である。

## 完了確認

`reviews/audit-completion.py` と `reviews/completion-audit.json` に終了時の監査を保存した。
3,126 stepsの連続性・有限な損失と勾配・8時間での終了、6 rankの一致、16 checkpointsの
重みとoptimizer stateのSHA-256、17サンプルファイルの全6問・token数・終了トークン、
全データファイルのhash、モデル・データrevision、元モデル対照の再現、200テストを確認した。
BF16の厳しい数値比較が不合格だったことも、この記録に明記している。

最終checkpointの使用例:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/uv.sh run --locked llm-jp-logkv infer \
  --adapter /mnt/raid0/llm-jp-logkv/checkpoints/distill-fixed-20261101T162321Z-5dfd0965/step-0003126 \
  --prompt '日本の首都はどこですか。短く答えてください。' --max-new-tokens 512
```

今回の実装・8時間試行・終了後評価は完了した。回答の取り違えとBF16経路差は残るため、
品質保持や実用性能を主張せず、次の設計・蒸留条件の比較材料とする。
