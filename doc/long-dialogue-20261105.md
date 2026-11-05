# LogKV長時間対話の観察（2026-11-05）

ユーザー依頼で、Codexを相手役として状態を保持した対話を進める。
目標はモデル自身が生成した累計100,000 tokens（analysis・EOSを含む）。
入力tokens・境界補完tokens・状態累計は別集計し、入力で水増しして目標達成とはしない。

- checkpoint: `distill-20261103/step-0006719`。固定revisionの元8BモデルとLoRA。
- GPU: 1。GPU 0の既存ユーザー用chat serverはそのまま利用可能。
- reasoning effort: medium、temperature: 0.8、top-p: 0.95、seed: 20261105。
- 毎ターンの生成上限3,072、最大実行時間7.5時間。学習や外部LLM API呼出しは行わない。
- 本実験: `/mnt/raid0/llm-jp-logkv/experiments/dialogue-100k-20261105-02`。

相手役のプロンプトは `prompts/000001.json` 等へ順に記録し、
`scripts/run_dialogue_experiment.py` が同じChatEngine・LogKV状態で処理する。
runner自体は相手役の返答を生成しない。Codexが応答を見て問いを調整する。
初期題材は架空の凪舟島での共同創作。灯台の信号、登場人物の関係、合言葉を導入し、
物語・計画・会話を進めながら過去の設定を確認する。問題文で答えを再提示した場合は
再導入として記録し、独立した想起成功とは数えない。

`transcript.md` は全文、各 `turns/NNNNNN/` は入力・出力token IDsと全応答。
`metrics.jsonl` はトークン数、時間、CUDA割当量・ピーク、圧縮状態tensor bytes。
`progress.json` と `current-turn.json` は現在地。約10,000状態tokensごとと終了時に
`states/*.pt` へ状態・RNGを保存する。`control.cmd` に `save_and_exit` を書くと停止する。
反復や設定混同を観測しても、結果を良く見せるために途中で状態をリセットしない。

事前検証で状態保存の入れ子に含まれる整数offsetの処理を修正した。
旧 `dialogue-100k-20261105-01` は記録系の試行として231生成tokensで停止し、
本実験には算入しない。CPUで状態を保存・読戻しした後の次token logitsの完全一致を確認。
`dialogue-smoke-20261105-02` は実8Bで8生成tokens・状態保存・正常終了を確認した。

機械的な長時間生成の成功と、文脈や事実を保持する能力は分けて評価する。
一般的なベンチマークや他モデルとの公平な比較として扱わない。
