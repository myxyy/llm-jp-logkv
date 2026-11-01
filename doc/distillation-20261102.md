# 固定減衰LogKVの蒸留（2026-11-02）

未蒸留の全層LogKVは初期4問で反復出力になったため、元モデルを教師とする
出力蒸留を実装した。今回のユーザー指示は「固定減衰・8時間・全6 GPU」であり、
通常の継続実験に対する最大2 GPUの制約は、この試行に限って明示的に変更された。
訓練の完了や品質の回復は、実行結果を確認してから判断する。

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

既存成果物を上書きしないため、run-dirとlog-dirは新しい名前にする。
これは今回承認された6 GPUのコマンドであり、将来の長時間実験の包括承認ではない。

```bash
./scripts/uv.sh run --locked python scripts/run_distillation.py \
  --gpus 0,1,2,3,4,5 \
  --log-dir /mnt/raid0/llm-jp-logkv/logs/NEW-RUN \
  --evaluate-final -- \
  --data /mnt/raid0/llm-jp-logkv/datasets/distill-cache-20261101T155341Z-4326ce39 \
  --run-dir /mnt/raid0/llm-jp-logkv/experiments/NEW-RUN \
  --max-hours 8 --sequence-length 1024 --batch-size 1 --grad-accum 4 \
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
計画的に止める場合はrun-dir内へ `STOP` ファイルを作ると、step境界で終了処理する。

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

8時間試行の実行先と結果は確認後に追記する。
