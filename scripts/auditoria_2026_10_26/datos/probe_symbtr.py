import sys, glob, collections, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/aj/Documentos/Relacionado con Escuela/Melodies-auditoria")
import pandas as pd, music21 as m21
from src.data.parsing import parse_score, extract_events
R="/home/aj/Documentos/Relacionado con Escuela/Melodies/"
cat=pd.read_csv(R+"artifacts/outputs/multicorpus_full/catalog/catalog.csv")
sym=cat[cat.source_name=="SymbTr"].path.tolist()
tot=collections.Counter(); files_micro=0
for p in sym[:100]:
    s=parse_score(R+p)
    ps=[pt for n in s.recurse().notes for pt in n.pitches]
    alters=collections.Counter((pt.accidental.name if pt.accidental else None, pt.microtone.cents) for pt in ps)
    micro=[pt for pt in ps if pt.microtone.cents!=0 or (pt.accidental is not None and pt.accidental.alter%1!=0)]
    if micro: files_micro+=1
    tot.update({k:v for k,v in alters.items() if k[1]!=0 or (k[0] not in (None,'natural','sharp','flat'))})
    tot["_pitches"]+=len(ps); tot["_micro"]+=len(micro)
print("SymbTr files:",len(sym),"files with microtonal pitches:",files_micro)
print(tot.most_common(15))
# show the pc mapping of a microtonal pitch
s=parse_score(R+sym[0]); ev=extract_events(s)
for n in s.recurse().notes:
    p=n.pitches[0]
    if p.microtone.cents!=0 or (p.accidental is not None and p.accidental.alter%1!=0):
        print("example:",p.nameWithOctave,"acc",p.accidental and p.accidental.name,"alter",p.accidental and p.accidental.alter,"microtone",p.microtone.cents,"ps",p.ps,"pitchClass",p.pitchClass,"midi",p.midi); break
