# Мерей зерттеуінің 01 деректер аудиті

Көлем: NonEnc/P2P × desktop/smartphone төрт dataset. Модель оқытылған жоқ.

Аудиттің PASS мәртебесі тек физикалық тексерулерге қатысты. Ғылыми дайындық: BLOCKED_FIRST_ONSET_HISTORY.

| Dataset | Raw | Retained | Positive users | Onset алдындағы negative тарихы бар user |
|---|---:|---:|---:|---:|
| NonEnc_desktop | 221145 | 218756 | 291 | 0 |
| NonEnc_smartphone | 221695 | 216457 | 167 | 0 |
| P2P_desktop | 204324 | 200658 | 220 | 0 |
| P2P_smartphone | 207956 | 200559 | 122 | 0 |

Timestamp START/END және timezone расталмаған. [t−1h,t+1h] аралығы — бекітілген шартты design; source фактісі ретінде расталмайды.
Алғашқы positive label нақты шабуыл басталған уақыт емес. Қайталанатын episode және future target бұл кезеңде есептелмейді.
Raw файлдар өзгертілмейді. Canonical mapping тек келесі жұмыстың ашық ережесі; басқа жобаның ranking PASS қорытындысы көшірілмеді.
Artifact пен кіріс hash-тары artifact_manifest.json арқылы --verify режимінде қайта тексеріледі.
