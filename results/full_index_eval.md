# F. Full-index evaluation (production path: e5-small-v2, dense + section router, airgapped)

Index: {'patient_notes': 300, 'medical_kb': 36466} | queries: 500 per variant

| variant | recall@1 | recall@5 | recall@10 | mrr@10 | p50 ms (tool call) | p95 ms |
|---|---|---|---|---|---|---|
| canonical | 0.928 | 0.992 | 0.994 | 0.957 | 22.73 | 37.03 |
| paraphrase | 0.856 | 0.962 | 0.974 | 0.904 | 20.14 | 30.31 |

## Misses at k=10

**canonical** (3): wrong topic: 3

**paraphrase** (13): wrong topic: 11; right topic, wrong section: 2

Examples:

- *what are the treatments for megdel syndrome ?* (gold: MEGDEL syndrome / treatment) → top: ('SCOT deficiency', 'treatment') — wrong topic
- *what is the outlook for glossopharyngeal neuralgia ?* (gold: Glossopharyngeal Neuralgia / outlook) → top: ('Syndactyly type 9', 'symptoms') — wrong topic
- *do you have information about blood count tests* (gold: Blood Count Tests / information) → top: ('Hemophilia', 'exams and tests') — wrong topic
- *Fraser like syndrome clinical presentation* (gold: Fraser like syndrome / symptoms) → top: ('Fraser syndrome', 'information') — wrong topic
- *ongoing clinical trials for Absence of the Septum Pellucidum* (gold: Absence of the Septum Pellucidum / research) → top: ('Childhood Central Nervous System Atypical Teratoid/Rhabdoid Tumor', 'research') — wrong topic
- *how would I know if someone has ICF syndrome* (gold: ICF syndrome / symptoms) → top: ('Idiopathic Pulmonary Fibrosis', 'exams and tests') — wrong topic
- *therapy options MEGDEL syndrome* (gold: MEGDEL syndrome / treatment) → top: ('Multifocal motor neuropathy', 'treatment') — wrong topic
- *Buerger disease clinical presentation* (gold: Buerger disease / symptoms) → top: ('Neuroblastoma', 'symptoms') — right topic, wrong section
- *prognosis of Glossopharyngeal Neuralgia* (gold: Glossopharyngeal Neuralgia / outlook) → top: ('Spondyloepimetaphyseal dysplasia x-linked with mental deterioration', 'symptoms') — wrong topic
- *therapy options enlarged parietal foramina* (gold: enlarged parietal foramina / treatment) → top: ('Prostate Enlargement: Benign Prostatic Hyperplasia', 'treatment') — right topic, wrong section
- *ongoing clinical trials for Chronic Pain* (gold: Chronic Pain / research) → top: ('Back Pain', 'research') — wrong topic
- *ongoing clinical trials for Herpes Zoster Oticus* (gold: Herpes Zoster Oticus / research) → top: ('Mycosis Fungoides and the Szary Syndrome', 'research') — wrong topic
