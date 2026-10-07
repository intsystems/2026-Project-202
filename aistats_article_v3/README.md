# Current deliverable: one complete manuscript

Open **aistats2027.pdf**. Main text and all appendices are now in one file. The optional blue PDF is an editorial copy of the same document. Build with `python build.py`; see SOURCE_LAYOUT.md and REVISION_2026_10_07_ru.md for the current changes.

The notes below describe earlier development stages.

# Новая версия AISTATS: теория, контролируемые системы, приложения

Самостоятельная переработка в новой папке. Название сохранено в формулировке из сообщения о поданной заявке:

**Detecting Simplification in Neural Network Dynamics from Scalar Logs: Active Dimension in Controlled Recurrent Systems**

## Файлы

Текущая редакция прошла дополнительную техническую и языковую вычитку по актуальному AISTATS PAT feedback: добавлены pipeline измерения и интерпретации MG, полная таблица scalar baselines, CUSUM и latency sensitivity, RL policy selection и forecast model selection, уточнены условия теоретической интерпретации, обозначения, протоколы и related work. Основной текст занимает 8 страниц; полный PDF — 27. Изменения по-прежнему обёрнуты в `\claude{...}`.

- `aistats2027.pdf` — обычная чёрная версия.
- `aistats2027_blue.pdf` — тот же текст с синей разметкой AI-assisted изменений.
- `MEASUREMENT.md`, `mg_pipeline.py`, `verify_pipeline.py` — standalone описание и пример измерения MG на новом scalar log с явными проверками unusable windows.
- `REPRODUCIBILITY.md`, `experiment_code/`, `measurement_code/`, `source_snapshot_manifest.json` — команды, снимок estimator/experiment source и SHA256 provenance.
- `aistats2027.tex` — основной редактируемый файл; текст разделов находится в `sections/`.
- `claim_sources.md` — происхождение чисел, единицы повторения и существенные различия постановок.
- `evidence/` — копии использованных числовых материалов и протоколов с SHA256.
- `build_validation.json`, `claims_validation.json` — результаты проверок.
- `aistats2027_v3_sources.zip` — PDF, исходники, рисунки, baseline-сравнения и числовые материалы для пересборки.

Все содержательные разделы переписаны. Основная линия: геометрическая интуиция из теории; её проверка на контролируемых режимах; независимая проверка относительного сигнала на задачах обучения; стоимость получения и обработки лога. Эксперименты разделены на **Controlled experiments** и **Applied experiments**. FORCE/E9 отнесены к оценке поведения обученных систем, а не к траектории весов при обучении.

В основном тексте подчёркнуты преимущества CNN-детектора, перенос его порога на заморозку ResNet, связь с информацией VAE, различение фаз и гармоник и стоимость относительно полного спектра Ляпунова. Сохранены важные границы: ReLU-death, непереносимость всех вмешательств на ResNet, результат циклического VAE и быстрые скалярные конкуренты. Утверждения «работает везде» и «быстрее всех подходов» не включены, поскольку сохранённые результаты их не подтверждают.

Найдены и включены MLP-контроли из прежней статьи. Они описаны с фактическими ошибками и ограничениями наблюдателя. Пример с точным безусловным восстановлением во всех режимах в найденных данных отсутствует. Совпадение оценки с covariance effective rank не выдаётся за доказательство точной active dimension.

## Сборка двух PDF

В корне проекта:

```powershell
..\.venv_walker\Scripts\python.exe aistats_article_v3/build.py
```

Команда пересобирает обе версии, проверяет предел восьми страниц основной части, ошибки ссылок и переполнения, равенство извлечённого текста и синий цвет в готовом PDF. Нужны Python, latexmk, pdfLaTeX/BibTeX и Poppler (`pdftotext`, `pdftohtml`). Стиль 2027 скопирован из ранее скачанного официального пакета. Шрифт и размеры основной полосы не изменялись; исправлена только избыточная фиксированная ширина служебного copyright box шаблона, чтобы он помещался в колонке.

Для регенерации рисунков и таблиц по текущим результатам проекта:

```powershell
..\.venv_walker\Scripts\python.exe aistats_article_v3/prepare_assets.py
..\.venv_walker\Scripts\python.exe aistats_article_v3/make_baseline_tables.py
..\.venv_walker\Scripts\python.exe aistats_article_v3/validate_claims.py
..\.venv_walker\Scripts\python.exe aistats_article_v3/build.py
..\.venv_walker\Scripts\python.exe aistats_article_v3/package.py
```

`prepare_assets.py` требует соседние экспериментальные папки и numpy/pandas/matplotlib. В распакованном архиве используйте `--bundled` для регенерации из приложенных числовых данных. Для обычной компиляции уже приложены готовые векторные рисунки и таблицы.

## Разметка изменений

Новый текст в `sections/` обёрнут в `\claude{...}`. По умолчанию он чёрный. Файл `aistats2027_blue.tex` включает синий режим и читает тот же основной источник. Используйте эту обёртку для дальнейших правок. Фигуры сохраняют собственную палитру; их новые подписи отмечены синим в blue-версии.

## Что осталось авторской проверкой перед подачей

AI Use Statement отражает указанное автором распределение ролей: основная научная постановка и финальные утверждения авторов, помощь ИИ в экспериментах и тексте, проверка результатов и дополнительная редактура авторами. Это не автоматическое свидетельство выполнения такой проверки. Checklist отвечает честно по наличным материалам: полный анонимизированный пакет обучения, полная опись оборудования и свод лицензий не заявлены как готовые. Архив этой версии воспроизводит таблицы, рисунки и PDF; он не содержит все обученные веса и исходные корпуса.

Этот источник не подключён к прежнему `build_all.py`: иначе сборка старой синхронизированной версии могла бы затереть самостоятельную переработку. Старые ICOMP, Artifacts и AISTATS файлы не изменялись. Материалы автоматически никуда не отправлялись.

## Revision of October 2, 2026

Removed machine-dependent timings from the abstract, revised headings and captions, clarified the dynamical-systems motivation, and replaced alarm terminology with detections and false positives. Checklist entries use explicit answers. Available hardware and software metadata are documented in the timing appendix and saved in `evidence/compute_provenance/`; missing historical hardware metadata are identified explicitly. Both PDF variants retain the AI-assisted revision markup.
