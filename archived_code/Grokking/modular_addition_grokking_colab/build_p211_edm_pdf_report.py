"""Build a self-contained Russian PDF report from the p=211 EDM analysis."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "edm_report_p211_wd05_tau1"
OUTPUT = ANALYSIS / "p211_wd05_detailed_edm_report_tau1_lb_mg.pdf"

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))

styles = getSampleStyleSheet()
BODY = ParagraphStyle(
    "BodyRu", parent=styles["BodyText"], fontName="Arial", fontSize=9.2,
    leading=12.1, alignment=TA_JUSTIFY, spaceAfter=2.4 * mm,
)
SMALL = ParagraphStyle(
    "SmallRu", parent=BODY, fontSize=7.8, leading=10, spaceAfter=1.5 * mm,
)
TITLE = ParagraphStyle(
    "TitleRu", parent=styles["Title"], fontName="Arial-Bold", fontSize=23,
    leading=28, textColor=colors.HexColor("#16253d"), alignment=TA_CENTER,
    spaceAfter=6 * mm,
)
SUBTITLE = ParagraphStyle(
    "SubtitleRu", parent=BODY, fontSize=11, leading=14, alignment=TA_CENTER,
    textColor=colors.HexColor("#41546f"), spaceAfter=8 * mm,
)
H1 = ParagraphStyle(
    "H1Ru", parent=styles["Heading1"], fontName="Arial-Bold", fontSize=16,
    leading=19, textColor=colors.HexColor("#163a68"), spaceBefore=2 * mm,
    spaceAfter=3.5 * mm,
)
H2 = ParagraphStyle(
    "H2Ru", parent=styles["Heading2"], fontName="Arial-Bold", fontSize=12,
    leading=15, textColor=colors.HexColor("#254f7d"), spaceBefore=2 * mm,
    spaceAfter=2 * mm,
)
CAPTION = ParagraphStyle(
    "CaptionRu", parent=SMALL, fontSize=7.4, leading=9.2, alignment=TA_LEFT,
    textColor=colors.HexColor("#45566e"), spaceBefore=1.2 * mm,
    spaceAfter=3 * mm,
)
CALLOUT = ParagraphStyle(
    "CalloutRu", parent=BODY, fontSize=10, leading=13,
    textColor=colors.HexColor("#17365d"), leftIndent=5 * mm, rightIndent=5 * mm,
    borderColor=colors.HexColor("#98b6d4"), borderWidth=0.8,
    borderPadding=4 * mm, backColor=colors.HexColor("#edf5fc"),
    spaceBefore=2 * mm, spaceAfter=4 * mm,
)
FORMULA = ParagraphStyle(
    "Formula", parent=BODY, fontName="Arial", fontSize=10.3, leading=14,
    alignment=TA_CENTER, leftIndent=5 * mm, rightIndent=5 * mm,
    backColor=colors.HexColor("#f4f6f8"), borderPadding=3 * mm,
    spaceAfter=3 * mm,
)


def P(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def page_header(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(14 * mm, 9 * mm, "EDM-анализ гроккинга: modular addition, p=211")
    canvas.drawRightString(A4[0] - 14 * mm, 9 * mm, f"стр. {doc.page}")
    canvas.restoreState()


def image(name: str, width=181 * mm, height=None) -> Image:
    path = ANALYSIS / name
    if not path.exists():
        # Fast/ablation analyses may intentionally skip the expensive E/k
        # sensitivity grid.  Keep the PDF build robust and explain omission
        # in the report rather than failing on a missing optional figure.
        return Spacer(1, 1 * mm)
    obj = Image(str(path))
    if height is None:
        height = width * obj.imageHeight / obj.imageWidth
    obj.drawWidth, obj.drawHeight = width, height
    return obj


def dimension_table(summary: pd.DataFrame, phase: str, metrics: list[str]) -> Table:
    labels = {
        "train_loss": "train loss", "val_loss": "validation loss",
        "weight_norm": "weight norm", "gradient_norm": "gradient norm",
        "update_norm": "update norm", "gradient_cosine": "gradient cosine",
        "gradient_participation_ratio": "gradient PR",
        "parameter_participation_ratio": "parameter PR",
    }
    subset = summary[(summary.series == "detrended") & (summary.phase == phase)]
    data = [[P("Метрика", SMALL), P("FNN", SMALL), P("Cao", SMALL),
             P("Simplex", SMALL), P("LB", SMALL), P("MG", SMALL)]]
    for metric in metrics:
        row = subset[subset.metric == metric].iloc[0]
        data.append([P(labels[metric], SMALL)] + [f"{row[m]:.2f}" for m in ("FNN", "Cao", "Simplex", "LB", "MG")])
    table = Table(data, colWidths=[51 * mm, 20 * mm, 20 * mm, 22 * mm, 22 * mm, 22 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Arial", 7.8),
        ("FONT", (0, 0), (-1, 0), "Arial-Bold", 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aebdca")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def build() -> None:
    global ANALYSIS, OUTPUT
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--output", type=Path, default=None)
    args, _ = parser.parse_known_args()
    ANALYSIS = args.analysis_dir
    OUTPUT = args.output or (ANALYSIS / f"{ANALYSIS.name}_detailed_edm_report_tau1_lb_mg.pdf")
    meta = json.loads((ANALYSIS / "transition_metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(ANALYSIS / "phase_dimension_summary.csv")

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=13 * mm, bottomMargin=15 * mm,
        title="Подробный EDM-анализ modular addition p=211, WD=0.5",
        author="Codex / 2026-Project-202",
        subject="EDM, intrinsic dimension, Levina-Bickel, MacKay-Ghahramani",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=page_header)])

    story = []
    story += [Spacer(1, 12 * mm), P("Подробный EDM-анализ гроккинга", TITLE),
              P("Modular addition, <i>p</i>=211 · OmniGrok-style Transformer · AdamW · fixed weight decay 0.5", SUBTITLE)]
    story += [P(
        "Отчёт по файлу <b>training_log_p_211_wd_0_point_5.csv</b>. "
        "Для всех реконструкций временного ряда зафиксировано <b>τ=1</b>. "
        "Основной sliding-window анализ использует W=200 строк лога (≈10 000 шагов), stride=10 строк "
        "(≈500 шагов), максимальную размерность задержанного вложения E=15 и k=5 ближайших соседей.",
        CALLOUT)]
    key_data = [
        ["Строк лога", str(meta["rows"])], ["Диапазон", f'{meta["first_step"]}–{meta["last_step"]} шагов'],
        ["Стабильная меморизация", f'{meta["stable_memorization_flag_step"]}'],
        ["Гроккинг", f'{meta["grok_flag_step"]}'], ["Gap по флагам", f'{meta["flag_gap_steps"]} шагов'],
        ["Gap по acc≥0.95", f'{meta["threshold_gap_steps"]} шагов'],
        ["Финальные accuracy", f'train={meta["final_train_acc"]:.3f}, val={meta["final_val_acc"]:.3f}'],
        ["Время обучения", f'{meta["elapsed_seconds"] / 60:.1f} мин'],
    ]
    table = Table(key_data, colWidths=[62 * mm, 75 * mm], hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Arial", 9), ("FONT", (0, 0), (0, -1), "Arial-Bold", 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f1f6fb"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b4c3d1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [table, Spacer(1, 6 * mm), P(
        "<b>Главный результат.</b> В логе есть выраженный режим delayed generalization: модель "
        "стабильно запоминает train около шага 13 650, а флаг genuine grokking возникает около 46 150. "
        "EDM-кривые не дают единственного истинного числа параметров», но показывают смену геометрии "
        "наблюдаемой динамики ещё до резкого роста validation accuracy.", CALLOUT), PageBreak()]

    story += [P("1. Траектория обучения и фазы", H1), image("01_training_overview.png", 181 * mm, 136 * mm),
              P("Рис. 1. Это <b>сырой (raw) тренировочный лог</b>: здесь показаны непосредственно train/validation loss, accuracy и weight norm; detrending и EDM к этим кривым не применяются. Синие/красные пунктирные линии — зафиксированные моменты stable memorization и genuine grokking. Между ними gap ≈32.5k шагов; по порогу accuracy≥0.95 gap ≈38.2k.", CAPTION),
              P("Train accuracy достигает 0.95 около шага 7 750 и 0.99 около 13 450. Validation accuracy остаётся почти нулевой большую часть обучения, достигает 0.5 лишь около 44 700 и затем быстро поднимается до 0.95 около 45 950. Одновременно weight norm после раннего максимума около 88 монотонно уменьшается примерно до 37 — характерный след сильной регуляризации и перехода к более простому решению."),
              P("Последние 5 000 шагов недостаточны для полноценного post-grok sliding-window окна W=10 000 шагов. Поэтому фазовые средние для post-grok отсутствуют: это свойство длины записи, а не ошибка расчёта."), PageBreak()]

    story += [P("2. Что именно измеряется", H1),
              P("Из каждого одномерного наблюдателя x(t) строится delay embedding:"),
              P("X<sub>t</sub> = [x(t), x(t−τ), …, x(t−(E−1)τ)], &nbsp; τ=1.", FORMULA),
              P("Размерность в отчёте — локальная геометрическая сложность облака таких векторов внутри окна. Это размерность реконструированной динамики конкретного scalar observer, а не число весов модели, не ранг матрицы слоя и не автоматически ранг LoRA. Один scalar observable может при корректной реконструкции кодировать многомерную динамику, но оценка зависит от наблюдаемости, длины окна, шума, кривизны, нестационарности и масштаба соседства."),
              P("Перед расчётом detrended-версии в каждом окне методом наименьших квадратов находится прямая x(i)=a·i+b и вычитается. После этого ряд стандартизуется. Так устраняется простое направленное движение среднего, чтобы оценка в большей степени отражала локальные степени свободы колебаний, а не один доминирующий тренд."),
              P("Использованы четыре проектных EDM-диагностики: (1) false nearest neighbours (FNN), (2) критерий Cao, (3) simplex projection и (4) kNN maximum-likelihood intrinsic dimension Levina–Bickel. Дополнительно на всех графиках размерности показана агрегация MacKay–Ghahramani."),
              P("Оценки FNN/Cao/Simplex имеют иной смысл и дискретную процедуру выбора E, поэтому их абсолютные значения нельзя считать взаимозаменяемыми с kNN-ID. Согласованное изменение нескольких методов важнее совпдения чисел."),
              P("Важно: соседние sliding windows перекрываются на 95%, поэтому кривые пригодны для диагностики эволюции, но точки на них не являются независимыми статистическими повторностями."), PageBreak()]

    story += [P("3. Levina–Bickel и MacKay–Ghahramani", H1),
              P("Для каждой точки X<sub>i</sub> delay-облака обозначим через T<sub>j</sub>(X<sub>i</sub>) расстояние до j-го ближайшего соседа. При локально постоянной плотности число точек в шаре моделируется пуассоновским процессом; рост объёма шара как r<sup>d</sup> позволяет восстановить d из отношений соседних радиусов."),
              P("d̂<sub>i</sub> = { (1/(k−1)) Σ<sub>j=1</sub><sup>k−1</sup> log[T<sub>k</sub>(X<sub>i</sub>)/T<sub>j</sub>(X<sub>i</sub>)] }<sup>−1</sup>", FORMULA),
              P("В используемой legacy-версии Levina–Bickel глобальное число получается арифметическим средним локальных оценок:"),
              P("d̂<sub>LB</sub> = (1/N) Σ<sub>i=1</sub><sup>N</sup> d̂<sub>i</sub>.", FORMULA),
              P("MacKay–Ghahramani предлагают сначала объединять аддитивные log-distance statistics, то есть усреднять обратные локальные оценки, и только затем инвертировать:"),
              P("d̂<sub>MG</sub> = { (1/N) Σ<sub>i=1</sub><sup>N</sup> d̂<sub>i</sub><sup>−1</sup> }<sup>−1</sup> = { [1/(N(k−1))] Σ<sub>i,j</sub> log[T<sub>k</sub>/T<sub>j</sub>] }<sup>−1</sup>.", FORMULA),
              P("Следовательно, MG численно является гармоническим средним локальных d̂<sub>i</sub>, но мотивация глубже простой «замены среднего ради устойчивости». В пуассоновской likelihood естественно складываются log-distance evidence / inverse-dimension statistics; инверсия каждой шумной локальной оценки до усреднения создаёт выпуклостный (Jensen) upward bias. Пулинг sufficient statistics выполняется раньше нелинейной инверсии и поэтому обычно меньше реагирует на редкие огромные локальные d̂."),
              P("Так как гармоническое среднее не превосходит арифметическое, MG систематически ниже LB. Это не две разные физические размерности и не доказательство того, что меньшая цифра «истиннее» в каждом окне; это две агрегации одного и того же локального kNN evidence с разными bias/variance свойствами."),
              P("В данном анализе k=5, E=15, евклидово расстояние, без Theiler exclusion. Последнее важно: соседние delay vectors временно коррелированы, поэтому абсолютные ID следует считать диагностическими. Для строгой оценки нужны Theiler window, несколько k, bootstrap по неперекрывающимся блокам и более длинная post-grok запись."), PageBreak()]

    story += [P("4. Все четыре EDM-метода на loss", H1), image("02b_four_methods_loss_raw_vs_detrended.png", 181 * mm, 101 * mm),
              P("Рис. 2. Верхний ряд — <b>RAW</b>-окна; нижний ряд — <b>DETREND</b>, где в каждом окне вычтена своя линейная регрессия. Это явное сравнение preprocessing для четырёх EDM-методов.", CAPTION),
              image("02_four_methods_loss_with_mg.png", 181 * mm, 101 * mm),
              P("Рис. 3. Все панели выше — <b>только detrended</b>. Raw и detrended оценки сохранены раздельно в tau1_edm_windows_all_methods.csv. Красная MG-ID validation loss — общий kNN-MLE reference.", CAPTION),
              P("Для validation loss MG-ID в основной части gap держится примерно в диапазоне 6–8, достигает максимума ≈8.14 около 24.5k, затем после ≈40k быстро падает: среднее 3.07 на 40k–46.15k и ≈2.09 на 45k–48k. LB показывает от же коллапс (примерно 9 → 4 → 3), но на более высоком уровне."),
              P("Это наиболее сильный ранний геометрический сигнал в отчёте: падение ID начинается за несколько тысяч шагов до formal grok flag и совпадает с уходом validation loss с плато. Однако это detection/concurrent precursor, а не доказанный причинный или универсально предсказывающий маркер."),
              P("Train loss ведёт себя иначе: MG-ID растёт примерно с 7.5 в раннем участке до 9.7–10.4 во время gap и не коллапсирует перед гроккингом. Поэтому train и validation observers видят разные стороны динамики; validation loss здесь действительно информативнее для перехода генерализации."), PageBreak()]

    story += [P("5. Численные фазовые оценки", H1),
              P("Средние detrended-оценки по окнам, полностью попадающим в memorization gap:"),
              dimension_table(summary, "memorization_gap", ["train_loss", "val_loss", "weight_norm", "gradient_norm", "update_norm", "gradient_cosine", "gradient_participation_ratio", "parameter_participation_ratio"]),
              Spacer(1, 4 * mm),
              P("В gap MG даёт: train loss 10.14, validation loss 6.60, weight norm 7.20, gradient norm 4.88, update norm 9.36, gradient cosine 12.32. LB соответственно выше: 13.12, 9.00, 10.12, 6.79, 12.24, 16.69."),
              P("При переходных окнах validation loss резко падает с MG 6.60 до 2.77 (−58%), weight norm — с 7.20 до 3.27 (−55%), тогда как gradient norm растёт с 4.88 до 5.98 (+22%) и update norm с 9.36 до 10.42 (+11%). Картина согласуется со сжатием наблюдаемой loss/weight геометрии при одновременной активизации градиентной динамики."),
              P("Simplex для train loss резко выше в gap (11.53), но для validation loss остаётся низким (3.16). Это расхождение подчёркивает, что simplex выбирает predictive embedding dimension, а kNN-MLE оценивает локальную геометрию; одно число нельзя подставлять вместо другого."),
              image("06_phase_heatmaps_lb_mg.png", 181 * mm, 91 * mm),
              P("Рис. 4. Фазовые средние LB и MG. Белая колонка post-grok — отсутствие полного окна после перехода, а не нулевая размерность.", CAPTION), PageBreak()]

    story += [P("6. Основные scalar observers", H1), image("03_core_metrics_lb_mg.png", 178 * mm, 239 * mm),
              P("Рис. 5. На каждом графике фиолетовая кривая — арифметическая LB, красная — pooled/harmonic MG. Все кривые здесь detrended; синяя вертикаль — memorization, красная — grokking.", CAPTION), PageBreak()]

    story += [P("7. Фиксированные случайные проекции", H1), image("04_projection_metrics_lb_mg.png", 174 * mm, 232 * mm),
              P("Рис. 6. Одномерные фиксированные random projections losses, weights, gradients и updates. Все размерности рассчитаны на detrended окнах; для компактности показан r0.", CAPTION), PageBreak()]

    story += [P("8. Raw против detrended", H1), image("05_raw_vs_detrended_lb_mg.png", 181 * mm, 120 * mm),
              P("Рис. 7. Прозрачные линии — raw, непрозрачные — после локального linear detrending. Разница показывает вклад монотонного дрейфа в геометрию окна.", CAPTION),
              P("Detrending не является нейтральным: он намеренно удаляет одну простую временную компоненту. Поэтому raw и detrended отвечают на разные вопросы. Raw ID описывает наблюдаемую траекторию целиком; detrended ID — остаточную локальную сложность вокруг лучшего линейного движения. Для задач раннего детектирования полезно логировать обе версии."),
              P("Нельзя интерпретировать detrended ID как прямое количество обучаемых весовых компонент. В частности, scalar validation loss является нелинейной функцией параметров и данных; разные parameter-space направления могут давать одинаковый loss, а одно направление — сложную временную кривую."), PageBreak()]

    story += [P("9. Робастность к длине окна", H1), image("07_window_robustness_lb_mg.png", 181 * mm, 120 * mm),
              P("Рис. 8. W∈{100,150,200,250} строк лога, то есть приблизительно 5k–12.5k optimizer steps. Пунктир — LB при W=200; оценки detrended.", CAPTION),
              P("Среднее MG меняется с W умеренно: validation loss ≈5.66→6.13, gradient norm ≈4.81→5.17, weight norm ≈5.68→6.15, train loss ≈8.75→9.63. Формы ключевых переходов сохраняются, но абсолютное значение возрастает с окном. Поэтому в сравнении экспериментов W, logging cadence, τ, E и k должны быть одинаковыми."),
              P("W=200 — компромисс: в delay cloud остаётся N=W−E+1=186 точек. Этого достаточно для дешёвой диагностики, но мало для точной высокой ID; особенно с k=5 и сильной временной корреляцией. Значения вблизи E=15 также потенциально насыщаются потолком embedding dimension."), PageBreak()]

    story += [P("10. Чувствительность к E и k", H1), image("08_mg_E_k_sensitivity.png", 181 * mm, 132 * mm),
              P("Рис. 9. MG-ID в заранее выбранном участке memorization gap для E∈{6,9,12,15}, k∈{5,8,10,15}; оценка выполнена после detrending.", CAPTION),
              P("Разброс по сетке значителен: train loss MG 5.10–10.96, validation loss 3.78–5.48, weight norm 4.14–6.67, gradient norm 3.53–5.51. Следовательно, абсолютную цифру нельзя представлять без гиперпараметров оценки. Более надёжны относительные изменения при фиксированном протоколе и устойчивость знака эффекта по нескольким E/k."),
              P("Рост k снижает variance, но усредняет более крупный и потенциально искривлённый участок многообразия; малый k лучше локализует геометрию, но чувствителен к шуму и конечной выборке. E должен быть достаточно большим для unfolding, однако слишком большой E при коротком окне ухудшает kNN distances и усиливает curse of dimensionality."), PageBreak()]

    story += [P("11. Интерпретация для проекта", H1),
              P("<b>Что поддерживается данными.</b> (i) Run демонстрирует genuine delayed generalization с большим gap. (ii) Геометрия validation loss резко упрощается перед/во время перехода: MG-ID падает приблизительно с 6–8 до 2, LB — с 8–11 до ≈3. (iii) Weight-norm ID тоже коллапсирует, тогда как gradient-norm ID растёт к переходу. (iv) MG воспроизводит основные формы LB, но даёт меньшие и менее подверженные локальным выбросам значения."),
              P("<b>Что пока не доказано.</b> Нельзя по одному seed утверждать универсальный predictor гроккинга; нельзя считать MG=6.6 «шестью активными параметрами»; нельзя напрямую выбрать LoRA rank=7. EDM одного scalar observable оценивает effective state-space complexity видимой динамики, а LoRA rank — размерность линейного подпространства обновления конкретной матрицы с требованием знать направления."),
              P("<b>Практическое применение.</b> Эти кривые полезны как дешёвые online phase indicators: детектор plateau → sustained ID shift → validation transition; как критерий сравнения optimizer/WD; как сигнал для adaptive checkpointing/early warning; как спооб выбрать моменты для дорогих parameter-space probes. Для layer-wise pruning/LoRA EDM лучше использовать только как proxy для бюджета, а направления и ранг проверять gradient/update sketches, randomized SVD или спектром ковариации обновлений."),
              P("Для честного прогноза следует обучить detector только на префиксе каждого run и оценивать lead time до заранее не использованного grok threshold на новых seeds/moduli. Сглаживание и окна нельзя центрировать через будущие точки в online-сценарии: текущий центрированный график использует данные с обеих сторон центра."),
              P("<b>Рекомендуемый следующий эксперимент:</b> 5–10 seeds при p=211 и два контрольных режима (не грокнувший и ранне-генерализующий), одинаковые W/τ/E/k, causal trailing windows, Theiler exclusion ≈E·τ, k∈{5,8,10,15}, bootstrap по блокам. Основной candidate score: стандартизованное падение MG-ID validation loss плюс рост MG-ID gradient norm; оценивать AUROC и median lead time."), PageBreak()]

    story += [P("12. Ограничения и воспроизводимость", H1),
              P("1. Один run и один seed; статистической оценки межзапусковой вариативности нет.<br/>"
                "2. Logging stride 50 шагов; τ=1 означает задержку именно в одну строку лога, то есть ≈50 optimizer steps.<br/>"
                "3. В delay embedding соседние точки перекрываются и временно коррелированы; Theiler exclusion не применён.<br/>"
                "4. Окна W=200 дают только 186 delay vectors при E=15.<br/>"
                "5. Сильная нестационарность нарушает локально-стационарные предпосылки kNN likelihood; detrending исправляет лишь линейную часть.<br/>"
                "6. Центрированные окна визуально могут создавать look-ahead; для реального мониторинга нужны trailing windows.<br/>"
                "7. Эффекты возле конца ряда оценены хуже и полноценного post-grok окна нет.<br/>"
                "8. Random projections r0 являются scalar sketches, а не layer-wise intrinsic rank."),
              P("Файлы воспроизводимости", H2),
              P("Скрипт анализа: <b>analyze_p211_wd05_edm.py</b><br/>"
                "Окна всех методов: <b>tau1_edm_windows_all_methods.csv</b><br/>"
                "Фазовые средние: <b>phase_dimension_summary.csv</b><br/>"
                "Window robustness: <b>window_robustness_lb_mg.csv</b><br/>"
                "E/k sensitivity: <b>E_k_sensitivity_lb_mg.csv</b><br/>"
                "Моменты переходов: <b>transition_metadata.json</b>"),
              P("Источники метода", H2),
              P("E. Levina, P. J. Bickel. <i>Maximum Likelihood Estimation of Intrinsic Dimension</i>, NeurIPS 2004/2005.<br/>"
                "D. J. C. MacKay, Z. Ghahramani. <i>Comments on ‘Maximum Likelihood Estimation of Intrinsic Dimension’</i>, 2005, inference.org.uk/mackay/dimension/.<br/>"
                "H. Kantz, T. Schreiber. <i>Nonlinear Time Series Analysis</i> — delay embeddings and nonlinear diagnostics.<br/>"
                "L. Cao. <i>Practical method for determining the minimum embedding dimension of a scalar time series</i>, Physica D, 1997."),
              P("Вывод", H2),
              P("Наиболее репрезентативный сигнал этого run — не абсолютная ID, а согласованный предгроккинговый коллапс LB/MG размерности validation loss и weight norm при росте сложности gradient dynamics. MacKay–Ghahramani логично использовать как основную kNN-MLE агрегацию, потому что она пулингует аддитивную likelihood-информацию до инверсии и уменьшает upward bias от экстремальных локальных оценок. При этом LB стоит сохранять рядом как диагностический индикатор неоднородности: большой разрыв LB−MG означает широкий/тяжёлохвостый набор локальных d̂."),
              Spacer(1, 6 * mm), P("Отчёт автоматически собран из CSV и артефактов анализа; дата сборки: 03.08.2026.", CAPTION)]

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
