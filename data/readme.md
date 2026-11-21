# Data

The repository includes [fast-mcts-kbqa-data.tar.zst](../fast-mcts-kbqa-data.tar.zst). Extract it once from the repository root. The archive preserves the `data/` directory structure.

```bash
tar --zstd -xf fast-mcts-kbqa-data.tar.zst
```

| Dataset | Evaluation file | Released questions | Sample IDs | Human annotations v2 |
| --- | --- | ---: | ---: | ---: |
| WebQSP | `WebQSP/test.json` | 1,621 | 500 | 100 |
| CWQ | `CWQ/test.json` | 500 | 500 | 200 |
| GrailQA | `GrailQA/dev.json` | 500 | 500 | 150 |
| KQA Pro | `KQAPro/test.json` | 900 | — | 450 |

Evaluation paths are relative to `data/`. WebQSP, CWQ, and GrailQA each have a `sample_info.json` containing their evaluation subset IDs. KQA Pro uses the fixed question IDs in `KQAPro/test.json`.

The 900 human-annotated training dialogues are in `data/human-anno-v2/{webqsp,cwq,grailqa,kqapro}/`, grouped by question type. The SFT script reads these JSON files directly.
