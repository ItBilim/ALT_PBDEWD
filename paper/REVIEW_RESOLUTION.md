# Аудит түзетулерінің орындалуы

Түпнұсқа PBDEWD.docx өзгертілмейді. Сақталған REVIEW.md бастапқы нұсқаның аудиті болып қалады. Жаңа PBDEWD_review_ready.docx — төртRQ-ға арналған әдістер мен нәтижелердің түзетілген көшірмесі.

- Төрт ΔAP толық precision-нен түзетілді: +0.1183,−0.0026,+0.0029,−0.0402.
- LaTeX literal формулалар native Word math түріне ауыстырылды; transfer minus және delta анықтамасы қалпына келтірілді; tied-score AP threshold арқылы түсіндірілді.
- R1 құрамындағы mean/activity/window-age ажыратылды; өлшемдер15/34/49/79/109 берілді.
- V2 exploratory provenance, original32historical reference және өзгермейтін40family нақты жазылды.
- User-cluster Poisson bootstrap,10000draw,pointwise/familywise айырмасы және conditional uncertainty шегі жазылды.
- Source envelope,historygap,EWMA,drift,static reference және shared cohort ережелері сипатталды.
- LR/HGB nestedHPO және held-out calibration әдісі қосылды; тек осы алгоритмдер жаңа пакетте үйретілді.
- S×A movement interaction түсіндірілді; factorial нәтижелерінен causal/statistical-confirmed claim жасалмайды.
- Raw→daily→FINAL көлемдері,UNSEEN өзhistory-і және уақыт шекаралары берілді.
- Device transfer16evaluation/8pairedcontrast,raw score және cross-risk scope нақты ажыратылды.
- ТөртRQ-ға қатысы жоқ3.10қосымша эксперименттер бөлімі алынып тасталды; ұсынылмаған appendix туралы claim жоқ.
- Кестелер1–7бірізді нөмірленді,қайталанатын header және row splitting ережесі қойылды.

Соңғы жаңа есептеу нәтижелері reference predictions және40CI-мен салыстырылады. Бет көрінісі рендер арқылы тексеріледі. Бұл corrections жаңа prospective validation немесе Q2 acceptance кепілдігі емес.
