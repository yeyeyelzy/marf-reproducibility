# Formal implementation parity

This release is a focused extraction of the code path used by the formal MARC
runs.  Unused exploratory modules were not carried into the release.  The
following behaviors are retained without changing the reported method:

| Formal component | Release component | Retained behavior |
| --- | --- | --- |
| TCEM embedding, graph, transition, and temporal attention | `marc/models/backbone.py`, `attention.py` | dimensions, initialization order, graph propagation, and visit aggregation |
| CCS ontology residual | `marc/models/hierarchy.py` | shared center/neighbor attention, trainable ancestor table, residual coefficient, and ontology hash binding |
| state-conditioned edge reweighting | `marc/models/edge.py` | endpoint/state MLP, sigmoid gate, and unchanged graph support |
| patient state and visit-gap encoding | `marc/models/temporal.py` | clipped 16-bin gaps and causal GRUCell state update |
| class-aware relation objective | `marc/models/relation.py` | ReLU cosine relations, class mask, prediction-sensitive target, and unordered off-diagonal pairs |
| optimizer and selection loop | `marc/training/engine.py` | Adam, staged learning rates, validation-AUROC selection, patience 5, and minimum improvement `1e-4` |
| MIMIC preprocessing | `marc/preprocessing/mimic.py` | cohort rules, held-out final visit, split-specific time weights, and training-patient graph scope |

The release uses strict checkpoint loading.  It records hashes of the private
adjacency and ontology in the private run manifest, and the hierarchy module
rejects a checkpoint produced with another ontology asset.  No private asset
or reported result is distributed in this repository.

Exact regeneration of a previously reported patient split requires the
private `split_ids.json` emitted by preprocessing, because the historical
splitter passed patient identifiers through Python sets.  A new authorized
preprocessing run is otherwise protocol-equivalent but is not guaranteed to
recover the identical patient membership under a different Python build.
