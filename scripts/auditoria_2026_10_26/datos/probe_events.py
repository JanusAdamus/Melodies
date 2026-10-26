import sys, collections
sys.path.insert(0,"/home/aj/Documentos/Relacionado con Escuela/Melodies-auditoria")
import music21 as m21
from src.data.parsing import parse_score, extract_events, select_parts
from src.data.observations import build_observation_sequence
LIB="/home/aj/Documentos/Relacionado con Escuela/Melodies/external/library/scores/"
for name in sys.argv[1:]:
    p = name if name.startswith("/") else LIB+name
    s=parse_score(p)
    parts=list(s.parts)
    print("=====",name)
    for i,pt in enumerate(parts):
        clefs=collections.Counter(type(c).__name__ for c in pt.recurse().getElementsByClass(m21.clef.Clef))
        notes=list(pt.flatten().notes)
        ties=sum(1 for n in notes if n.tie is not None and n.tie.type in ("stop","continue"))
        grace=sum(1 for n in notes if n.duration.isGrace)
        cs=sum(1 for n in notes if isinstance(n,m21.harmony.ChordSymbol))
        voices=len(pt.recurse().getElementsByClass(m21.stream.Voice))
        print(f" part{i} {type(pt).__name__} {pt.partName!r} clefs={dict(clefs)} notes+chords={len(notes)} tie_continuations={ties} grace={grace} chordsymbols={cs} voices={voices}")
    for pt_flag in (False,True):
        ev=extract_events(s,prefer_treble=pt_flag)
        sel=select_parts(s,prefer_treble=pt_flag)
        print(f" prefer_treble={pt_flag}: parts_used={len(sel)} n_events={len(ev)}")
    ev=extract_events(s,prefer_treble=False)
    print(" first 8 events:",[(e.offset,e.label,e.measure,e.beat) for e in ev[:8]])
    nm=[e for e in ev if e.measure is None]; print(" events with measure None:",len(nm))
    try:
        o=build_observation_sequence(ev,"pitch_class"); print(" obs ok",o.size)
    except Exception as exc: print(" obs ERROR",type(exc).__name__,exc)
    bad=[e for e in ev if e.representative_pitch_class is None]; print(" events w/o rep pc:",[(e.offset,e.kind,e.label,e.measure) for e in bad[:5]])
