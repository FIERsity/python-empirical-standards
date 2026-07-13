# Data validation and panel diagnostics

## Validated merges

Every merge should declare its expected key relationship: `one_to_one`, `one_to_many`,
`many_to_one`, or `many_to_many`. `merge_validated` rejects violated cardinality and missing
keys, reports unmatched rows on both sides, and can require complete matching from either
input. Many-to-many merges are permitted only when explicitly requested because they can
multiply observations.

```python
merged = merge_validated(
    outcomes,
    city_attributes,
    on="city_id",
    relationship="many_to_one",
    require_all_left=True,
)
```

The returned `MergeResult` contains both the merged frame and a `MergeReport`. Store the
report with analysis outputs rather than relying on a console-only merge message.

## Panel diagnostics

`diagnose_panel` reports duplicate and missing keys, balance, coverage, observations per
entity, singleton entities, time coverage, and variance decompositions for declared numeric
variables. Variables with zero within-entity variation are flagged as absorbed by entity
fixed effects; variables with zero within-time variation are flagged as absorbed by time
fixed effects.

The variance decomposition is diagnostic rather than a hypothesis test. Very small within
variation deserves substantive inspection even when it is not exactly zero. A balanced
panel is not automatically preferable: deleting observations solely to force balance can
change the target population and induce selection.

