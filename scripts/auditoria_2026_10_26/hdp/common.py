import csv, os, sys
import numpy as np
REPO = "/home/aj/Documentos/Relacionado con Escuela/Melodies-auditoria"
DATA = "/home/aj/Documentos/Relacionado con Escuela/Melodies"
sys.path.insert(0, REPO)
from src.data.parsing import extract_events, parse_score
from src.data.observations import build_observation_sequence
ANALYSIS = DATA + "/artifacts/outputs/multicorpus_full/analysis/analysis.csv"

def rows():
    return [r for r in csv.DictReader(open(ANALYSIS)) if r["hdp_log_likelihood"]]

def load(path):
    p = path if os.path.isabs(path) else os.path.join(DATA, path)
    return build_observation_sequence(extract_events(parse_score(p), prefer_treble=False), "pitch_class")
