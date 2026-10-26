import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/aj/Documentos/Relacionado con Escuela/Melodies-auditoria")
import music21 as m21
from src.data.parsing import parse_score, merge_parts, extract_events
from src.data.observations import build_observation_sequence
L="/home/aj/Documentos/Relacionado con Escuela/Melodies/external/library/scores/"
s=parse_score(L+"Happy_Birthday_To_You_C_Major.mxl")
m=merge_parts(s,prefer_treble=False)
print("HappyBirthday elements at offset 1.0:",[(type(e).__name__, getattr(e,'figure',None), [p.nameWithOctave for p in e.pitches]) for e in m.flatten().notesAndRests if e.offset==1.0])
s=parse_score(L+"Hungarian_Dance_No_5_in_G_Minor.mxl")
m=merge_parts(s,prefer_treble=False)
print("Hungarian empty chords:",[(type(e).__name__, getattr(e,'figure',None), e.measureNumber) for e in m.flatten().notes if isinstance(e,m21.chord.Chord) and not e.pitches])
s=parse_score(L+"Sonate_No._14_Moonlight_3rd_Movement.mxl")
ev=extract_events(s)
st=[n for n in s.recurse().notes if n.tie is not None and n.tie.type=="stop"][:3]
print("Sonata tie-stop examples (counted as new events):",[(n.measureNumber,n.offset,n.pitches[0].nameWithOctave) for n in st])
stripped=s.stripTies()
print("Sonata events: pipeline",len(ev),"after stripTies",len(extract_events(stripped)))
g=[e for e in ev if e.duration==0.0]; print("Sonata zero-duration (grace) events:",len(g))
# vocab mismatch finite vs hdp for non pitch_class obs
s=parse_score(L+"Bach_Minuet_in_G_Major_BWV_Anh._114.mxl"); ev=extract_events(s)
for o in ("pitch_class","midi_note","interval","pitch_class_duration"):
    ob=build_observation_sequence(ev,o); print(o,"vocab",ob.vocab_size,"len",ob.size)
print("pitch_class_duration bin of 0.75 ->", __import__('src.data.observations',fromlist=['x'])._duration_to_bin(0.75,(0.25,0.5,1.0,2.0,4.0)), "| 1.5 ->",__import__('src.data.observations',fromlist=['x'])._duration_to_bin(1.5,(0.25,0.5,1.0,2.0,4.0)), "| 3.0 ->",__import__('src.data.observations',fromlist=['x'])._duration_to_bin(3.0,(0.25,0.5,1.0,2.0,4.0)), "| 0.0 ->",__import__('src.data.observations',fromlist=['x'])._duration_to_bin(0.0,(0.25,0.5,1.0,2.0,4.0)))
