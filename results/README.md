# Төрт зерттеу сұрағының нәтижелері

| Сұрақ | Негізгі кесте | Түсіндірме |
|---|---|---|
| RQ1 history/activity қосымша құндылығы | RQ1_contrasts.csv | 8FINAL UNSEEN R1−R0 effect,pointwise және40-family CI |
| RQ2 толық profile қосымша құндылығы | RQ2_contrasts.csv | 8FINAL UNSEEN R4−R1;4оң/4теріс,барлық95%CIzero |
| RQ3 компоненттік contrast | RQ3_factorial_effects.csv | S/A/D marginal differences;S×A movement;CI жоқ |
| RQ4 device transfer | RQ4_device_transfer.csv | 8paired contrast,16source/model/representation evaluation;raw AP,CI жоқ |

main_metrics.csv және factorial_metrics.csv — барлық retained main/factorial metric. main_predictions.parquet және factorial_predictions.parquet — exact row identity бар болжамдар. exploratory_40_and_descriptive_contrasts.csv — толық40family және DEV/KNOWN descriptive results. uncertainty_metadata.json — user-cluster bootstrap тәртібі.

fresh_refit_reference_checks.json және result_contract.json — fresh fitting нәтижесінің алдыңғы numerical evidence-мен сәйкес болуы. test_summary.json/test_receipt.txt —27тест. preparation/*.json —4raw dataset-тен exactfeature rebuild. final_gate.json/artifact_manifest.json — sealed hashes; техPASS ғылыми positive benefit емес.
