"""Prepara el corpus local del smoke (58 archivos) a una cache JSONL, sin tocar el repo."""
import sys
from dataclasses import replace
from next_token_experiment.protocol import build_default_experiment_config
from next_token_experiment.data.preprocess import prepare_corpus
cfg = build_default_experiment_config()
cfg = replace(cfg, corpus=replace(cfg.corpus, root_dir=sys.argv[1]))
prep = prepare_corpus(cfg, max_files=None, n_workers=4, cache_path=sys.argv[2])
print(len(prep.pieces), len(prep.exclusions))
