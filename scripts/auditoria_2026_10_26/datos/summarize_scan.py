import pandas as pd
d=pd.read_csv("scan_corpus.csv")
print("files",len(d),"parse errors",d.get("err",pd.Series()).notna().sum())
for src,g in d.groupby("source"):
    print(f"== {src}: n={len(g)}")
    for col in ["chordsymbols","empty_chords","tie_cont","grace","microtonal_pitches","repeats"]:
        print(f"  {col}: files>0={int((g[col]>0).sum())} total={int(g[col].sum())}")
    multi=g[g.n_parts>1]
    print(f"  multi-part files={len(multi)}; prefer_treble still keeps >1 part in {(multi.parts_treble>1).sum()} of them")
    print(f"  events all parts={int(g.n_events_all.sum())} vs treble-only={int(g.n_events_treble.sum())} ({1-g.n_events_treble.sum()/g.n_events_all.sum():.1%} of events come from non-treble parts)")
    bad=g[g.max_measure_num!=g.measures_part0]; print(f"  files where max measure number != measure count: {len(bad)}")
print(d[d.file.str.contains("Sonate_No._14_Moonlight_3rd")][["n_events_all","tie_cont","grace","measures_part0","max_measure_num"]].to_string())
