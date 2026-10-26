import sys, warnings, collections; warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/aj/Documentos/Relacionado con Escuela/Melodies-auditoria")
import pandas as pd, music21 as m21
from src.data.parsing import parse_score, extract_events
from src.data.observations import build_observation_sequence
print("half-flat / half-sharp -> pitchClass vs midi (music21 rounding):")
for step in "CDEFGAB":
    for acc in ("half-flat","half-sharp"):
        p=m21.pitch.Pitch(step+"4"); p.accidental=m21.pitch.Accidental(acc)
        print(f"  {step}{acc:10s} ps={p.ps:5.1f} pitchClass={p.pitchClass:2d} midi%12={p.midi%12:2d}")
R="/home/aj/Documentos/Relacionado con Escuela/Melodies/"
cat=pd.read_csv(R+"artifacts/outputs/multicorpus_full/catalog/catalog.csv")
seen=collections.Counter()
for p in cat[cat.source_name=="SymbTr"].path:
    s=parse_score(R+p)
    for n in s.recurse().notes:
        for pt in n.pitches:
            if pt.accidental is not None and pt.accidental.name in ("slash-flat","half-flat","half-sharp","slash-quarter-sharp"):
                k=(pt.step,pt.accidental.name,pt.accidental.alter,pt.ps,pt.pitchClass,pt.midi%12)
                seen[k]+=1
for k,v in seen.most_common(): print(" step,acc,alter,ps,pitchClass,midi%12:",k,"count",v)
