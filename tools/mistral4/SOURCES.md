# Sources of these tools

Each file is a copy of the tool that produced Petit Chonk's records, with only its local paths replaced
(`publish_tools.py`; the substitutions are the only differences). The original path is in the work
folder the records name.

"This file's SHA-256" is of the file as stored in the repository, with LF line endings. A checkout
that converts line endings (Git's `core.autocrlf` on Windows, and `.gitattributes`' `*.bat eol=crlf`)
gives other bytes; `git show HEAD:<path> | sha256sum` gives the stored ones.

| File | Original | Original SHA-256 | This file's SHA-256 | Substitutions |
|---|---|---|---|---|
| `fti/make_sequences.py` | `scripts/fti/t3/make_sequences.py` | `553865d943159132…` | `8b1fd2d13d7f5663…` | 4 |
| `fti/ref_logits.cpp` | `scripts/fti/t4/ref_logits.cpp` | `c66897fa173a71a7…` | `c66897fa173a71a7…` | 0 |
| `fti/build_ref_logits.bat` | `scripts/fti/t4/build_ref_logits.bat` | `2a3387cbd340fdaf…` | `4c8831af1748bff9…` | 2 |
| `fti/t4_runs.py` | `scripts/fti/t4/t4_runs.py` | `137df20a468e3502…` | `fc5f811447ec2fdc…` | 3 |
| `fti/t6_measure.py` | `scripts/fti/t6/t6_measure.py` | `a5b9be024d68c66d…` | `a5b9be024d68c66d…` | 0 |
| `fti/t6_contract_eval.py` | `scripts/fti/t6/t6_contract_eval.py` | `1ff59e522fa8f261…` | `3b06433f885f4f12…` | 2 |
| `fti/t6_rule.py` | `scripts/fti/t6/t6_rule.py` | `f34a77b24e986205…` | `d2930f06acc5c235…` | 2 |
| `fti/t8_run.py` | `scripts/fti/t6/t8_run.py` | `67ce2228fe8f618b…` | `10523fe48bf9f38b…` | 3 |
| `fti/thresholds_UD-Q4_K_M.json` | `fti/t6/thresholds_UD-Q4_K_M.json` | `e5ea6dfb14fe0679…` | `e5ea6dfb14fe0679…` | 0 |
| `bench/fetch_sources.py` | `scripts/pub/p4/fetch_sources.py` | `81325ebc9f0f169f…` | `81325ebc9f0f169f…` | 0 |
| `bench/fetch_mmlu_yaml.py` | `scripts/pub/p4/fetch_mmlu_yaml.py` | `7306f19bbb4c6234…` | `7306f19bbb4c6234…` | 0 |
| `bench/freeze_a.py` | `scripts/pub/p4/freeze_a.py` | `72e78aaedbd796f5…` | `b4986686247a7232…` | 1 |
| `bench/check_render.py` | `scripts/pub/p4/check_render.py` | `1fdce2789b5dae28…` | `1fdce2789b5dae28…` | 0 |
| `bench/ref_logprobs.cpp` | `scripts/pub/p4/ref_logprobs.cpp` | `ba2a3278ba04db87…` | `ba2a3278ba04db87…` | 0 |
| `bench/build_ref_logprobs.bat` | `scripts/pub/p4/build_ref_logprobs.bat` | `8ca1079208fd827a…` | `1f7cedc9b1b5c581…` | 3 |
| `bench/p4_cand.py` | `scripts/pub/p4/p4_cand.py` | `0e4f1e846a6b739d…` | `2fcd008b1cd804a4…` | 1 |
| `bench/p4_ref.py` | `scripts/pub/p4/p4_ref.py` | `b23e70b3d94e96b1…` | `522b9f666583747e…` | 3 |
| `bench/p4_score.py` | `scripts/pub/p4/p4_score.py` | `21ed5ffc30b2f250…` | `21ed5ffc30b2f250…` | 0 |
| `bench/p4_qual.py` | `scripts/pub/p4/p4_qual.py` | `b4097ad18bcd5eb2…` | `b4097ad18bcd5eb2…` | 0 |
| `bench/p4_qual_runs.py` | `scripts/pub/p4/p4_qual_runs.py` | `1eff0a574079699d…` | `d4cb6adbcb31b4d6…` | 5 |
| `bench/p5_runs.py` | `scripts/pub/p5/p5_runs.py` | `27560ebec435d642…` | `2d46f77ddffad35f…` | 4 |
| `bench/test/stub_engine.py` | `scripts/pub/p4/test/stub_engine.py` | `cda338a2fab86412…` | `cda338a2fab86412…` | 0 |
| `bench/test/test_p4_cand.py` | `scripts/pub/p4/test/test_p4_cand.py` | `e962ac207d214c75…` | `e962ac207d214c75…` | 0 |
