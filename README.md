# PBDEWD төрт зерттеу сұрағы

Бұл пакетте мақаладағы RQ1–RQ4 үшін қажетті нақты эксперимент қана қалдырылды. Нәтижелер V2 ретроспективті exploratory дизайнын қайта орындайды; жаңа тәуелсіз растау емес.

Негізгі қорытынды: history/activity блогы 8 салыстырудың 7-еуінде жоғары AP көрсетеді; толық жеке профильдің R1-ден сенімді қосымша артықшылығы анықталмады. R4−R1 сегіз pointwise95% интервалдың бәрі нөлді қамтиды. Бұл equivalence немесе профиль ешқашан пайдасыз деген дәлел емес.

## Рецензентке кіру нүктелері

- Түзетілген әдістер мен нәтижелер: [PBDEWD_review_ready.docx](paper/PBDEWD_review_ready.docx). Бұл толық introduction/literature/discussion бар мақала емес, тек 3–4 бөлімдердің рецензенттік нұсқасы.
- Түпнұсқа аудит: [REVIEW.md](publication_analysis/2026-11-02/manuscript_review/REVIEW.md), өзгеріссіз сақталған.
- Аудит түзетулерінің орындалуы: [REVIEW_RESOLUTION.md](paper/REVIEW_RESOLUTION.md).
- Жоспар: [protocol/plan.json](protocol/plan.json).
- Соңғы тексеру: [results/final_gate.json](results/final_gate.json).
- Төрт RQ кестелері: [results](results/README.md).

## Папкалар

| Папка | Мақсаты |
|---|---|
| data/raw | Төрт бастапқы Parquet, өзгертілмеген |
| data/mapping | Аудиттен өткен raw row retention mapping |
| data/prepared | Қайта құрылып салыстырылған daily features және predictor metadata |
| experiment | Feature, split, train-only preprocessing, HPO/calibration, factorial, bootstrap, transfer коды |
| models/main | LR/HGB × 4dataset × 5fold × R0–R4 =200 saved pipeline |
| models/factorial | LR/HGB × 4dataset × 5fold × 8mask =320 saved pipeline |
| results | Қайта есептелген RQ1–RQ4 нәтижелері, болжамдар, тексеру журналдары |
| protocol/historical | Бұрынғы32 және V2 40 statistical reference; ағымдағы модельдермен шатастырылмайды |
| protocol/reference | Қайта орындауды бұрынғы дәлелмен салыстыруға арналған шектеулі reference predictions |
| paper | Түзетілген Word және оны құрастыратын editable JSON/code |
| tests | Future/label/user leakage және AP/bootstrap correctness тесттері |
| runtime | Windows Python3.12 үшін ғылыми кітапханалар; эксперимент тармағы емес |

## Қайта орындау

Python3.12.14, нақты нұсқалар requirements.txt ішінде. Windows bundled кітапханалары runtime/python_packages ішінде, numpy/pandas Python runtime-да болуы керек. Басқа жүйеде бөлек virtual environment ішінде requirements.txt бойынша орнатыңыз; басқа платформадағы exact execution бұл пакетпен сертификатталмаған.

```powershell
./run_experiment.ps1 -Mode Verify
./run_experiment.ps1 -Mode Prepare
./run_experiment.ps1 -Mode All
./run_experiment.ps1 -Mode Tests
./run_experiment.ps1 -Mode Seal
```

Басқа Python жолы: `$env:PBDEWD_PYTHON='C:/path/to/python.exe'`. PowerShell-сіз: `python -B -m experiment.run --all`, `python -B -m experiment.tests_run`.

All жарамды checkpoint-ті қайта қолданады. Осы жеткізілген пакеттегі520 модель fresh fit арқылы жасалған; оған2400 inner HPO trial қосылады, барлығы2920 estimator fit және520 бөлек calibrator. Trials estimator-лері сақталмайды, selection predictions/parameters/AP ledger сақталады. Models файлын басқа нұсқамен үнсіз алмастырмаңыз. Aggregate не Tests нәтижелерін қайта жазғаннан кейін Seal қажет; Verify ештеңе fit етпейді және файл өзгертпейді.

Pickle тек осы hash арқылы бекітілген жергілікті артефактілер үшін ашылады. Native score және calibrated score ажыратылған. Негізгі statistical family40 өзгермейді, мақалаға8contrast көрсету түзетуді8-ге қысқартпайды.

## Ғылыми шектер

R1 тек mean емес:15current +15past mean +3history activity +window age =34. R0/R2/R3/R4 өлшемдері15/49/79/109. Factorial қозғалыс блогы тек S және A бірге болғанда қосылады. UNSEEN өз өткен label-free history-ін пайдаланады; zero-history cold-start емес.

Бұл completed-window risk ranking. Verified incident onset, қалыпты pre-onset history, continuous coverage және exact source clock/window generator толық расталмаған. Early warning/lead time/episode recall және independent external validation туралы claim жоқ. Төрт файл бір released population-ға жатады.

RF/XGB/IF, trajectory/CP, history sensitivity, policy, ескі00–19 orchestrator және уақытша build/QA файлдары белсенді пакетке кірмейді. Бастапқы32 нәтиже тек historical reference ретінде сақталады, оларды жаңа prospective растау деп ұсынбаймыз.
