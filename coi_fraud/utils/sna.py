
from collections import defaultdict
import pandas as pd

def sna_light(df: pd.DataFrame):
    out_sum = df.groupby("persona_1", observed=True)["monto"].sum().to_dict()
    in_sum = df.groupby("persona_2", observed=True)["monto"].sum().to_dict()
    df["p1_out_sum"] = df["persona_1"].map(out_sum).fillna(0.0).astype(float)
    df["p2_in_sum"]  = df["persona_2"].map(in_sum).fillna(0.0).astype(float)

    adj_out = defaultdict(set)
    und = defaultdict(set)
    for a,b in zip(df["persona_1"], df["persona_2"]):
        adj_out[a].add(b); und[a].add(b); und[b].add(a)
    nodes = set(df["persona_1"]).union(set(df["persona_2"]))
    in_triangle = {n: False for n in nodes}
    in_cycle = {n: False for n in nodes}

    for n in nodes:
        nbrs = list(und.get(n, []))
        for i in range(len(nbrs)):
            u = nbrs[i]; Nu = und.get(u, set())
            for j in range(i+1, len(nbrs)):
                v = nbrs[j]
                if v in Nu: in_triangle[n] = True; break
            if in_triangle[n]: break

    for a in nodes:
        for b in adj_out.get(a, []):
            for c in adj_out.get(b, []):
                if a in adj_out.get(c, set()):
                    in_cycle[a] = in_cycle[b] = in_cycle[c] = True
                for d in adj_out.get(c, []):
                    if d != a and b != d and a in adj_out.get(d, set()):
                        in_cycle[a] = in_cycle[b] = in_cycle[c] = in_cycle[d] = True

    df["p1_in_triangle"] = df["persona_1"].map(in_triangle)
    df["p2_in_triangle"] = df["persona_2"].map(in_triangle)
    df["p1_in_cycle"]    = df["persona_1"].map(in_cycle)
    df["p2_in_cycle"]    = df["persona_2"].map(in_cycle)
    return df
