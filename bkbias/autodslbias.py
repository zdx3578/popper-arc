
with open("classDSLresult2.json") as f:
    dsl_map = json.load(f)

body_preds = []
arity = {}
for sig, funs in dsl_map.items():
    in_types, out_type = eval(sig)
    for fun in funs:
        arity[fun] = len(in_types) + 1    # +1 for GridId
        body_preds.append(fun)
        