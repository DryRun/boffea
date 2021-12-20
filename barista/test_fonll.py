from coffea.lookup_tools import extractor
ext = extractor()
ext.add_weight_sets(["* * sfs.json"])
ext.finalize()
evaluator = ext.make_evaluator()

print("available evaluator keys:")
for key in evaluator.keys():
    print("\t", key)
print(evaluator["FONLL_SF/pt_value"])