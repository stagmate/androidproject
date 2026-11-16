# F. Full-index evaluation (production path: e5-small-v2, dense + section router, airgapped)

Index: {'patient_notes': 300, 'medical_kb': 35811} | queries: 500 per variant

| variant | recall@1 | recall@5 | recall@10 | mrr@10 | p50 ms (tool call) | p95 ms |
|---|---|---|---|---|---|---|
| canonical | 0.884 | 0.948 | 0.950 | 0.913 | 19.71 | 28.44 |
| paraphrase | 0.862 | 0.968 | 0.980 | 0.911 | 18.86 | 27.29 |

## Misses at k=10

**canonical** (25): wrong topic: 25

**paraphrase** (10): wrong topic: 9; right topic, wrong section: 1

Examples:

- *is childhood myocerebrohepatopathy spectrum inherited ?* (gold: Nijmegen breakage syndrome / inheritance) → top: ('childhood myocerebrohepatopathy spectrum', 'frequency') — wrong topic
- *is spondyloepimetaphyseal dysplasia, strudwick type inherited ?* (gold: congenital stromal corneal dystrophy / inheritance) → top: ('spondyloepimetaphyseal dysplasia, Strudwick type', 'information') — wrong topic
- *is fraser syndrome inherited ?* (gold: Nijmegen breakage syndrome / inheritance) → top: ('Gordon syndrome', 'inheritance') — wrong topic
- *is ankyloblepharon-ectodermal defects-cleft lip/palate syndrome inherited ?* (gold: congenital stromal corneal dystrophy / inheritance) → top: ('ankyloblepharon-ectodermal defects-cleft lip/palate syndrome', 'information') — wrong topic
- *what are the treatments for megdel syndrome ?* (gold: MEGDEL syndrome / treatment) → top: ('SCOT deficiency', 'treatment') — wrong topic
- *is mevalonate kinase deficiency inherited ?* (gold: Nijmegen breakage syndrome / inheritance) → top: ('deoxyguanosine kinase deficiency', 'inheritance') — wrong topic
- *is juvenile primary osteoporosis inherited ?* (gold: congenital stromal corneal dystrophy / inheritance) → top: ('juvenile primary osteoporosis', 'information') — wrong topic
- *is alg12-congenital disorder of glycosylation inherited ?* (gold: Nijmegen breakage syndrome / inheritance) → top: ('ALG12-congenital disorder of glycosylation', 'information') — wrong topic
- *what is the outlook for glossopharyngeal neuralgia ?* (gold: Glossopharyngeal Neuralgia / outlook) → top: ('Syndactyly type 9', 'symptoms') — wrong topic
- *what research (or clinical trials) is being done for intraocular (uveal) melanoma ?* (gold: Hypopharyngeal Cancer / research) → top: ('Intraocular (Uveal) Melanoma', 'treatment') — wrong topic
- *is arginine:glycine amidinotransferase deficiency inherited ?* (gold: Nijmegen breakage syndrome / inheritance) → top: ('arginine:glycine amidinotransferase deficiency', 'information') — wrong topic
- *is pyridoxal 5'-phosphate-dependent epilepsy inherited ?* (gold: Nijmegen breakage syndrome / inheritance) → top: ("pyridoxal 5'-phosphate-dependent epilepsy", 'information') — wrong topic
