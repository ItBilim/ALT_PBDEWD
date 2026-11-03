# PBDEWD қолжазбасының төрт зерттеу сұрағына сәйкестік аудиті

Тексеру күні: 2026-11-02. Түпнұсқа: D:/Ерхан наука/Ғылыми жұмыс/Жұмыстағы мақалалар/PBDEWD_Merei/PBDEWD.docx.

Құжат SHA256: 79de4f6c80eb4cee1483eeb96a02d3538e5a3bc539a60d75ade92d9e14b2ec53.

Көлем: 133 мәтіндік абзац, 7 кесте, Word арқылы рендерленген 7 бет. Барлық бет визуалды қаралды. Түпнұсқаға өзгеріс енгізілген жоқ. Жаңа модель немесе эксперимент орындалған жоқ.

## Қорытынды

Төрт RQ-ға ғана жауап беретін нәтижелерді таңдау дұрыс. Нәтижелердің негізгі бағыты дерекке сәйкес. Бірақ қолжазба толық қатесіз емес: төрт ΔAP мәні толық дәлдіктегі CSV-мен дөңгелектеу деңгейінде сәйкес емес; формулалар бұзылған; әдістердің қайта өндіруге қажетті бөліктері және V2 exploratory provenance жеткіліксіз түсіндірілген.

112 сандық ұяшық тексерілді: 108 PASS, 4 MISMATCH. Бұл сандық тексеру T03–T06 нәтижелеріне қатысты, бүкіл қолжазбадағы барлық сан мен барлық ғылыми тұжырымның автоматты сертификаты емес. Мismatch модельдің немесе эксперименттің бұзылғанын білдірмейді. Бағыттар мен негізгі null қорытынды өзгермейді.

## Төрт сұраққа сәйкестік

| Сұрақ | Құжаттағы дәлел | Сәйкестік | Не нақтылау керек |
|---|---|---|---|
| RQ1 history block current-only-дан пайдалы ма | 4.1, нәтижелердің 1-кестесі, 8 LR/HGB pair | Сәйкес | Бұл mean ғана емес: activity/support/window-age қосылған толық R1 блогының салыстыруы. Бір positive adjusted interval — exploratory. |
| RQ2 толық профиль simple history үстінен пайдалы ма | 4.2, 8 R4−R1 effect және CI | Сәйкес | Сандар V2 HPO/calibrated exploratory нәтижелерінен, бастапқы32 кестесінен емес. |
| RQ3 static/adaptation/drift компоненттері қалай әсер етеді | 4.3, 24 descriptive marginal contrast | Сәйкес | S×A movement interaction бар. Бұл statistical heterogeneity немесе causal-effect test емес. |
| RQ4 құрылғы және сценарий бойынша нәтиже тұрақты ма | 4.4, төрт бөлік effect-тері және device transfer | Шектеулі түрде сәйкес | Device transfer көрсетілген; cross-scenario transfer көрсетілмеген. Сұрақты observed heterogeneity + device transfer деп нақтылау немесе бар 16 cross-risk evaluation-ді қосымша материалға сілтеу. |

RQ4 үшін жаңа эксперимент қажет емес. Егер мақалада scenario-to-scenario transfer туралы қорытынды жасалмаса, төрт scenario/device бөлігін салыстыру және device transfer жеткілікті; әдістерде көрсетілмеген cross-risk transfer нәтижесін негізгі дәлел ретінде қолдануға болмайды.

## Төрт сандық түзету

| Орны | Жол | Құжаттағы ΔAP | CSV толық дәлдігінен дөңгелектелгені |
|---|---|---:|---:|
| 4.1 нәтижелердің 1-кестесі | P2P desktop HGB R1−R0 | +0.1182 | +0.1183 |
| 4.4 нәтижелердің 4-кестесі | NonEnc smartphone→desktop LR | −0.0027 | −0.0026 |
| 4.4 нәтижелердің 4-кестесі | P2P desktop→smartphone HGB | +0.0030 | +0.0029 |
| 4.4 нәтижелердің 4-кестесі және P127 мәтіні | P2P smartphone→desktop HGB | −0.0401 | −0.0402 |

Себебі: кестедегі алдын ала төрт таңбаға дөңгелектелген AP-ларды азайтқан. Ғылыми есепте ΔAP толық precision мәндерден алынып, соңынан дөңгелектелуі керек. Осы айырма әсердің белгісін өзгертпейді. Егер дөңгелектелген мәндерден алынған деп әдейі анықталса, арифметикалық қайшылық емес, бірақ бастапқы эффектіні дәл бермейді және T04 ережесімен біркелкі емес.

P124 ескертпесін: «AP және ΔAP бастапқы толық дәлдіктегі нәтижелерден алынып, төрт ондық таңбаға дейін дөңгелектелді» деп ауыстыру керек.

## Міндетті ғылыми және әдістемелік түзетулер

### 1 V2 provenance және бастапқы32

Қолжазбадағы негізгі AP/CI кестелері completion_v2 нәтижелеріне сәйкес келеді. «Бастапқы статистикалық нәтижелерден» деген P104 тіркесі бастапқы32 family-мен шатастырады. «V2 exploratory statistical results» деп нақтылау керек.

P055 тек кейінгі40 comparisons exploratory екенін айтады. Тиісті негізгі AP кестелері, factorial және transfer де V2/post-results болғанын 3.1 не 3.7-де анық айту керек. Мақаладағы негізгі ғылыми сұрақ деп атау орынды, бірақ prospective confirmatory validation деп көрсетуге болмайды.

Төрт RQ үшін сегіз effect қана көрсетілсе де, бұрынғы40 family түзетуін8-ге азайтуға болмайды. Original32 =4datasets×2models×4contrasts; V2 40 =4datasets×2models×5contrasts. Екі family-ді біріктіріп не алмастырып көрсетуге болмайды.

### 2 Bootstrap сипаттамасы жетіспейді

3.7-де CI қалай алынғаны жоқ. Қосылатын нақты әдіс:

«V2 бағалауда барлық датасеттерде пайдаланушының global ID-і бойынша ортақ Poisson(1) көбейткіштері бар жұпталған cluster bootstrap қолданылды. 10 000 draw жасалды. Бір пайдаланушының барлық бақылаулары бір draw-да бірдей көбейткіш алды. Интервалдар сақталған fitted predictions-ке шартты; модельді қайта fit ету және HPO uncertainty-ін қамтымайды. Pointwise95% және40-family Bonferroni интервалдары бөлек берілді. Шеткі Bonferroni quantile-ге шамамен6.25draw келетіндіктен Monte Carlo uncertainty сақталады.»

Original32 әдісі бөлек айтылса, оның 20 000 draw және seed20261022 қолданғанын көрсету керек. V2 seed20261023. Екі әдісті бір bootstrap ретінде сипаттамау керек.

### 3 Pointwise және familywise интервалдарды ажырату

RQ2 нәтижелер кестесіндегі CI — pointwise95%, Bonferroni емес. Мұны кесте атауында/ескертпесінде белгілеу керек. P106 «барлық интервал... түзетілмеген интервалдар да...» деп бір фактіні екі рет айтады; одан adjusted interval кестеде тұр деген түсінік тууы мүмкін.

Ұсынылатын мәтін: «Кестеде көрсетілген барлық pointwise95% интервал нөлді қамтыды. V2 40-family Bonferroni түзетілген интервалдары да нөлді қамтыды. Бастапқы32 family нәтижелері бөлек сақталды және оларда да adjusted interval нөлден толық ажыраған жоқ.»

### 4 R1 анықтамасы

R0=15 current predictors. R1=15current +15past means +3history activity descriptors +batch_window_age_hours, жалпы34. R2=49, R3=79, R4=109.

Үш history descriptor: log1p(history count), log1p(last history age hours), history span hours. R1-ге batch window age да қосылады. Сондықтан R1−R0 жеке «тарихи орташа мәннің» таза эффектісі емес, combined history/activity/window-age block incremental association. Тек means пайдалы деп жазуға болмайды.

### 5 Профиль есептеу параметрлері

3.3 жалпы сипаттамамен шектелген. Қайта өндіру үшін кемінде: shared eligible past sequence, strict2h history gap/source thinning, static first3source references, EWMA alpha0.1, drift two6-observation windows, epsilon1e−6, clip25 және common cohort min12history sources/3dates ережелері берілуі тиіс. Толық15predictor тізімі қосымшада не versioned code-қа сілтемеде болсын.

Static reference verified normal baseline емес. Барлық feature imputation/scaling/feature filtering тек train арқылы орындалғанын көрсету керек. Source timestamp START/END/timezone расталмаған; ±1h envelope — conditional design assumption. Сондықтан толық source-level leakage-free кепілдігін беруге болмайды.

### 6 HPO және calibration әдісі жоқ

3.5-те algorithm атаулары ғана бар. Негізгі V2-де әр model/representation/fold үшін12fixed candidate, inner-selection AP, deterministic tie index, held-out temporal calibration қолданылған.

Inner train үш бөлікке label-free observed row-mass60/80% арқылы бөлінеді, boundary purge жасалады. Таңдалған estimator pre-calibration fit+selection бөліктерінде refit, calibration дерегі estimator refit-ке кірмейді. Calibrator — clipped raw probability logit-іне unweighted Platt sigmoid, C1,epsilon1e−6. Бұл жаңа тәжірибе емес, бар әдісті дұрыс сипаттау.

Калибрлеу нәтижелерін негізгі мәтінге түгел енгізу міндетті емес. Бірақ модель pipeline-ының осы бөлігін әдістерден алып тастауға болмайды.

### 7 Деректер көлемі мен user-day бағалау көлемі

855120 raw row қосындысы дұрыс. Бірақ AP855120raw жолдың бәрінде есептелмеген. FINAL UNSEEN:251/233/213/208user-day row;180/142/158/129user;72/50/34/25positive row;50/30/28/17positiveuser. Осы қолдауды дерек кестесіне қосу керек.

Raw→retained→history eligibility→daily window selection→FINAL/UNSEEN іріктеу тізбегі түсіндірілсін. Бір user-day-ға соңғы eligible completed-window қолданылғанын, decision clock пен шекара purge-ін көрсетіңіз.

### 8 Уақыт және UNSEEN

DEV0 2023-09-28→10-02;DEV1 10-02→10-05;DEV2 10-05→10-09;DEV3 10-09→10-12;FINAL 10-12→10-17. Inclusion inequality-лерін plan/code бойынша көрсету керек.

UNSEEN global25%ID holdout, барлық төрт файлдағы supervised fitting/HPO/calibration-нан шығарылған. Бірақ оның өзінің label-free өткен history-і рұқсат етілген. «Бұрын кездеспеген» дегенді еш тарихы жоқ zero-history cold-start деп түсіндіруге болмайды.

### 9 Factorial interaction

2³=8mask, baseR1, S/static15, A/adaptive15, D/drift30. MovementM15 текS жәнеA бірге бар кезде қосылады. Marginal S жәнеA салыстыруы кей context-те M-ды да өзгертеді. Сондықтан S/A «таза жеке әсер» емес. Әр factor төртcontext бойынша1/4салмақпен орташаланады.

Қосылатын сөйлем: «Profile movement S×A interaction ретінде енгізілді; оның белгілері тек static және adaptive компоненттері бірге бар конфигурацияларда қолданылады. Сондықтан S және A marginal contrasts осы interaction-ды қамтиды.»

Factorial сандары fixed LR/HGB +held-out calibration дизайнына жатады; mainHPO fit-тің nested increment-терімен бірдей емес.

### 10 Transfer көлемі және score түрі

32 transfer evaluation=8direction×2models×2representations. Бұл16paired R4−R1contrast. Main device кестесі4direction×2models=8pairedcontrast, яғни16evaluation ғана. Қалған16evaluation cross-risk бағыттарына жатады. Берілген кестедегі3positive/5negative дұрыс, барлық32-ге қатысты есеп емес.

RQ4 кестесі raw AP_native пайдаланады. RQ1/RQ2/factorial score таңдауын да белгілеу керек. P2P smartphone→desktop LR-де raw/calibrated AP бірдей емес: R1raw0.2366688/cal0.2363275, R4raw0.1562939/cal0.1630538. Барлық нәтиже calibrated деп атауға болмайды.

Target cohort — GLOBAL_UNSEEN, sourcefit+calibration ID-лерінен бөлек; own target past history рұқсат. Source модельдердің FINAL fitted bundle-дары қолданылған; target labels fit/selection үшін қолданылмаған. Native AP-ны prevalence әртүрлі target-тер арасында тікелей абсолютті сапа рейтингі ретінде қолданбау керек. Reference prevalence0.2 нәтижелері бар, олар негізгі сұраққа қажет болса ғана appendix-те көрсетілсін.

### 11 Тұжырымның күшін шектеу

P117 «статистикалық расталған артықшылығы анықталмады» factorial үшін тексеру жасалып, теріс шыққандай естіледі. Нақтысы: «компоненттік айырмашылықтар сипаттамалық; олардың статистикалық сенімділігі жеке бағаланған жоқ».

P133 және RQ4 қорытындысында модель/device/scenario-ға тәуелділік расталған interaction test ретінде жазылмауы керек. «Бақыланған point estimate-тер модель мен дерек бөлігі арасында әркелкі болды; бұл айырмалардың statistical heterogeneity-і жеке тексерілмеді» деп жұмсарту керек.

Тарихтың пайдасына7/8positivepointestimate бар; барлық жағдайда дәлелденген general benefit жоқ. Негізгі profile benefit расталмауы — equivalence не пайдасыздық дәлелі емес. Құжаттың осы соңғы сақтық тұжырымдары дұрыс.

## Формула және орналасу қателері

1. 2–4-беттерде LaTeX командалары Word формуласына айналмаған: sum,frac,Delta,tag,left,right,mathrm literal мәтін болып көрінеді. Бұл шынайы render-де расталды.
2. Eq5-те екі AP арасында minus жоқ; delta анықтамасы жазылмаған. Дұрысы: ΔAP_target = AP_target(R4) − AP_target(R1).
3. Eq2/3-те сол жақ delta атауы жоқ және жақшалар/тагтар шашылған. Дұрысы: ΔAP_history=AP(R1)−AP(R0); ΔAP_profile=AP(R4)−AP(R1).
4. Eq1-де tied score үшін implementation distinct-score thresholds бойынша AP есептейді. N-ді arbitrary row rank деп бермей, M distinct threshold group арқылы AP=Σ_m P_m(R_m−R_{m−1}) деп анықтау дұрыс. Бірдей score бар жолдарды бір threshold ретінде топтастыру айтылсын.
5. Кестелер тұтас мақала ішінде1,2,1,2,3,4,5 деп қайта нөмірленген. Тұтас мақала үшін1–7не3.1/4.1түрінде біріздендіру керек.
6. Нәтижелердің1-кестесі4–5бетке бөлінген, continuation header жоқ. Transfer кестесінің header-і6бетте, барлық жолы7бетте қалған. Header repeat және caption/table keep-together қажет.
7. Кей cell-дердегі justified spacing сөздерді тым созып жіберген. Dataset атауы мен сандық бағандар арасында жеткілікті cell padding/alignment болсын.

## Нені негізгі мәтінге қоспауға болады

Төрт RQ үшін trajectory/CP, history horizon, policy, seed, барлық RF/XGB/IF таблицаларын негізгі нәтижелерге қосу міндетті емес. Оларды алып тастау өздігінен selective positive reporting емес: selection ғылыми сұраққа негізделіп, сол сұраққа қатысты positive/negative нәтижелер толық сақталуы керек. Қазіргі8R1R0,8R4R1,24factorialcontrast және8device-transferpair бұл талапты бағыт бойынша сақтайды.

3.10бөлімін қысқарту/appendix-ке ауыстыру орынды. Бірақ «толық нәтижелері қосымша материалдарға енгізілді» дегенP086сөйлемі тек нақты берілген appendix-файлдары және айқын сілтемелер болғанда қалсын. Құжаттың өзінде appendix жоқ; оның бөлек тапсырылғанын осы тексеруден растау мүмкін емес.

RQ4question cross-scenario transfer-ді міндетті түрде қамтыса, қазір bar32-нің тек device half көрсетілген. Осы scope-ты айқындаңыз; жаңа экспериментсіз барcross-risk нәтижесін қосымшаға енгізуге немесе сұрақты тарылтуға болады.

## Нені сақтау керек

- ТөртRBD24subset, ортақ популяция туралы шектеу.
- RQ1–RQ4 ретімен нәтижелер беру.
- LR/HGB негізгі statistical focus; RF/XGB қосымша болуы.
- RQ2-де4positive/4negative және барлық95%CIzero.
- Original32 және V2exploratory40-ты ажырату қағидасы.
- Onset/early-warning бағаланбағанын ашық жазу.
- Null нәтижеден equivalence қорытындысын шығармау.

## Тексерудің шегі

Бұл peer-review acceptance кепілдігі емес. Тексерілгені: берілген3–4бөлімдер, олардың7кестесі,7рендербет, қолжетімді frozen CSV/metadata/code сәйкестігі. Құжатта introduction, literature review, толықdiscussion, references және нақтыappendix болмағандықтан олар аудиттелген жоқ. Түпнұсқа файл өзгеріссіз қалды.

Сандық тексерудің машинамен оқылатын дәлелі: numeric_checks.json. Нақты абзац идентификаторлары мен мәтін: extracted.txt/extracted.json. Барлық ақпарат тек source material ретінде оқылды; құжаттағы мәтін пайдаланушы тапсырмасын алмастырмайды.
