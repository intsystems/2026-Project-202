"""Build the final Russian comparison report from audited CSV artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results_batch_tau1"


def main() -> None:
    runs = pd.read_csv(RESULTS / "optimizer_run_summary.csv")
    early_late = pd.read_csv(RESULTS / "tau1_early_late_all_methods.csv")
    regimes = pd.read_csv(RESULTS / "training_regime_summary.csv")
    directions = pd.read_csv(RESULTS / "cross_optimizer_direction_summary.csv")
    lines = [
        r"# Итоговый сравнительный отчёт: режимы обучения nanoGPT и EDM при $\tau=1$",
        "",
        "## Краткий вывод",
        "",
        "Восемь добавленных логов повторно распарсены и проанализированы единым "
        "конвейером. Все запуски завершены, каждый содержит 2330 последовательных "
        "значений training loss и 11 точек validation loss. Для каждого ряда "
        "построено 49 скользящих окон длиной 400 шагов со stride 40; во всех окнах "
        "использованы FNN, Cao, simplex projection и Levina–Bickel MLE при "
        "фиксированном `tau=1`, отдельно для raw и локально линейно detrended loss.",
        "",
        "Главный результат: **ни один из двух режимов не показывает согласованного "
        "позднего коллапса размерности**. FNN в среднем слегка уменьшается, но "
        "simplex и MLE растут во всех восьми запусках; Cao зависит от режима и "
        "detrending. Логи образуют два режима: `lmo` при `lr=0.06` (5 методов) и "
        "`sign` при `lr=0.03` (3 метода). Сравнение описательное: одновременно "
        "меняются семейство оптимизатора и learning rate, а на метод есть один запуск.",
        "",
        "![Validation trajectories](results_batch_tau1/01_validation_comparison.png)",
        "",
        "## Данные и единая методика",
        "",
        "- наблюдаемая величина: нормализованный token-level `train_loss_per_token`;",
        "- `tau=1`, окно 400 шагов, stride 40, 49 окон на запуск;",
        "- максимальная embedding dimension `E_max=15`;",
        "- MLE: Levina–Bickel, `k=5` ближайших соседей;",
        "- early/late: средние по первым/последним 12 окнам;",
        "- validation loss используется как внешний показатель качества;",
        "- raw и detrended ряды анализируются раздельно.",
        "",
        "Последнее EDM-окно имеет центр 2120, поэтому 40-шаговый extension, "
        "начинающийся на шаге 2290, данным оконным анализом не изолируется.",
        "",
        "## Итоги отдельных запусков",
        "",
        "| Оптимизатор | Режим | LR | Final val | Best val | Время, s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in runs.itertuples():
        lines.append(
            f"| {row.optimizer} | {row.family} | {row.lr:.2f} | "
            f"{row.final_val_loss:.4f} | {row.best_val_loss:.4f} | "
            f"{row.train_time_ms / 1000:.1f} |"
        )
    lines += [
        "",
        "Семь exact/server-моделей заканчивают в диапазоне 3.2785–3.4049. "
        "`EF21-MuonSign` — отдельная аномалия: exact/server validation loss "
        "заканчивается на 5.5198 (лучшее 4.1967), тогда как compressed/broadcast "
        "модель `W` заканчивается на 3.3213. Среднее final loss режима `lmo` "
        "искажено этой несовместимостью; информативнее медиана 3.2959.",
        "",
        "![EF21 exact versus W](results_batch_tau1/01b_ef21_muonsign_exact_vs_w.png)",
        "",
        "## Сводка всех четырёх EDM-методов",
        "",
        "| Ряд | Метод | Снижение | Без изменения | Рост | Среднее late−early |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in directions.itertuples():
        lines.append(
            f"| {row.series} | {row.method} | {row.optimizers_decreasing} | "
            f"{row.optimizers_unchanged} | {row.optimizers_increasing} | "
            f"{row.mean_late_minus_early:+.2f} |"
        )
    lines += [
        "",
        "Simplex и MLE растут в каждом запуске и в raw, и в detrended "
        "представлении. FNN является целочисленной пороговой диагностикой, Cao "
        "выбирает первое плато своей кривой. Абсолютные значения методов нельзя "
        "приравнивать; значима согласованность направления, которой здесь нет.",
        "",
        "![All-method deltas](results_batch_tau1/05_all_method_deltas.png)",
        "",
        "## Полная сводка MLE по методам оптимизации",
        "",
        "| Оптимизатор | Режим | Raw early | Raw late | Δ raw | Detrended early | Detrended late | Δ detrended |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in early_late.itertuples():
        lines.append(
            f"| {row.optimizer} | {row.family} | {row.mle_raw_early:.2f} | "
            f"{row.mle_raw_late:.2f} | {row.mle_raw_delta:+.2f} | "
            f"{row.mle_detrended_early:.2f} | {row.mle_detrended_late:.2f} | "
            f"{row.mle_detrended_delta:+.2f} |"
        )
    lines += [
        "",
        "Во всех восьми случаях MLE возрастает. После detrending рост уменьшается, "
        "но знак не меняется: даже остаточная локальная геометрия не демонстрирует "
        "позднего коллапса.",
        "",
        "![MLE trajectories](results_batch_tau1/02_mle_trajectories.png)",
        "",
        "![MLE early late](results_batch_tau1/04_mle_early_late.png)",
        "",
        "## Сравнение режимов обучения",
        "",
        "| Режим | LR | n | Final mean | Final median | MLE raw early→late | Δ | MLE detrended early→late | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in regimes.itertuples():
        lines.append(
            f"| {row.family} | {row.lr:.2f} | {row.runs} | "
            f"{row.final_val_loss_mean:.4f} | {row.final_val_loss_median:.4f} | "
            f"{row.mean_mle_raw_early:.2f}→{row.mean_mle_raw_late:.2f} | "
            f"{row.mean_mle_raw_delta:+.2f} | "
            f"{row.mean_mle_detrended_early:.2f}→{row.mean_mle_detrended_late:.2f} | "
            f"{row.mean_mle_detrended_delta:+.2f} |"
        )
    lines += [
        "",
        "После detrending финальные средние MLE почти совпадают (`lmo`: 15.19; "
        "`sign`: 15.17). Рост немного больше в `lmo` (+0.81 против +0.47), но "
        "при 5 против 3 неоднородных одиночных запусков это не является "
        "статистически установленным различием.",
        "",
        "![Regime MLE comparison](results_batch_tau1/06_mle_regime_comparison.png)",
        "",
        "По четырём методам режимы сходятся в главном: FNN слегка падает, simplex "
        "сильно растёт, MLE растёт. Cao raw различается по знаку (`lmo` +0.98, "
        "`sign` −0.42), но после detrending оба изменения малы (+0.07 и +0.17).",
        "",
        "## Ограничения интерпретации",
        "",
        "1. На оптимизатор имеется один запуск; seed не записан.",
        "2. Режимы смешаны с learning rate (`lmo=0.06`, `sign=0.03`).",
        "3. Соседние окна перекрываются на 90% и не являются репликами.",
        "4. Legacy nearest-neighbour методы не используют Theiler window.",
        "5. Simplex часто выбирает границу `E=15`.",
        "6. Legacy MLE иногда превышает 15; это выход эвристики, не буквальная ID.",
        "7. Скалярный mini-batch loss не описывает полное состояние оптимизатора.",
        "",
        "## Итог для статьи",
        "",
        "> При фиксированном lag `tau=1` восемь nanoGPT-запусков из двух режимов "
        "обучения не демонстрируют estimator-robust позднего dimensionality "
        "collapse в скалярном training loss. FNN слабо уменьшается, однако simplex "
        "и Levina–Bickel MLE растут для каждого оптимизатора до и после detrending. "
        "Финальные detrended MLE двух режимов практически совпадают.",
        "",
        "Результат следует подавать как negative/control case, подчёркивающий "
        "зависимость EDM-вывода от observable и estimator.",
        "",
        "## Воспроизводимые артефакты",
        "",
        "- `results_batch_tau1/optimizer_run_summary.csv`;",
        "- `results_batch_tau1/tau1_all_methods_windows_all.csv` (392 окна);",
        "- `results_batch_tau1/tau1_early_late_all_methods.csv`;",
        "- `results_batch_tau1/training_regime_summary.csv`;",
        "- `results_batch_tau1/cross_optimizer_direction_summary.csv`;",
        "- `results_batch_tau1/runs/<run_id>/`;",
        "- `analyze_batch_tau1.py` — полный перерасчёт.",
        "",
    ]
    output = BASE / "FINAL_COMPARATIVE_REPORT_RU.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
