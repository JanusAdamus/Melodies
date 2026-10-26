import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/aj/Documentos/Relacionado con Escuela/Melodies-auditoria")
import pandas as pd, music21 as m21
from src.data.parsing import parse_score, extract_events, select_parts
ROOT="/home/aj/Documentos/Relacionado con Escuela/Melodies/"
cat=pd.read_csv(ROOT+"artifacts/outputs/multicorpus_full/catalog/catalog.csv")
rows=[]
for _,r in cat.iterrows():
    d={"source":r.source_name,"file":r.filename}
    try:
        s=parse_score(ROOT+r.path)
    except Exception as e:
        d["err"]=type(e).__name__; rows.append(d); continue
    notes=list(s.recurse().notes)
    d["chordsymbols"]=sum(isinstance(n,m21.harmony.ChordSymbol) for n in notes)
    d["empty_chords"]=sum(isinstance(n,m21.chord.Chord) and not n.pitches for n in notes)
    d["tie_cont"]=sum(1 for n in notes if n.tie is not None and n.tie.type in("stop","continue") and not isinstance(n,m21.harmony.ChordSymbol))
    d["grace"]=sum(1 for n in notes if n.duration.isGrace)
    pitches=[p for n in notes if not isinstance(n,m21.harmony.ChordSymbol) for p in n.pitches]
    d["microtonal_pitches"]=sum(1 for p in pitches if p.microtone.cents!=0 or (p.accidental is not None and p.accidental.alter%1!=0))
    d["n_pitches"]=len(pitches)
    d["n_parts"]=len(s.parts)
    d["n_events_all"]=len(extract_events(s,prefer_treble=False))
    d["parts_treble"]=len(select_parts(s,True))
    d["n_events_treble"]=len(extract_events(s,prefer_treble=True))
    d["repeats"]=len(list(s.recurse().getElementsByClass(m21.bar.Repeat)))
    ms=list(s.parts[0].getElementsByClass(m21.stream.Measure)) if s.parts else []
    d["measures_part0"]=len(ms); d["max_measure_num"]=max([m.number for m in ms],default=0)
    rows.append(d); print(d, flush=True)
pd.DataFrame(rows).to_csv("scan_corpus.csv",index=False)
