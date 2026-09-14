# Claim 2: Property-Inference Resistance

MARIO reduces sensitive-property inference from the server-exposed representation.

- **Paper reference:** Figure 5
- **Run:** `./claims/claim2/run.sh`
- **Observed result:** `claims/claim2/results/metrics.csv`
- **Expected result:** `claims/claim2/expected/metrics.csv`
- **Validation:** Compare the Male and Young attack accuracies and confirm that MARIO reduces exploitable sensitive-attribute information, with Male accuracy approaching the majority-class baseline.
