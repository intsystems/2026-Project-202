# MG для выбора модели прогноза рекуррентной активности

## Идея

В обученных рекуррентных генераторах заранее неизвестно, какую модель прогноза выбрать: периодическую модель, если система почти циклическая, или авторегрессию, если динамика содержит несколько независимых компонентов. MG по одному scalar trace может служить дешёвым сигналом для этого выбора.

## Постановка

Использованы сохранённые 1,000-unit FORCE generators: T1--T4, H2, H4, M4 и chaotic control. Наблюдается activity neuron 0. Из первых 4,096 samples строятся два прогноза на следующие 512 samples:

- Fourier model с 8 гармониками и частотой, выбранной по scalar recurrence;
- ridge autoregression с 64 лагами.

Выбор модели делается по MG, spectral entropy, normalized increments или scalar recurrence. Thresholds и directions подбираются на pilot seeds 1--2 и замораживаются. Основной результат оценивается на seeds 3--5. Дополнительно проверены два более поздних временных участка тех же trajectories.

## Результат

На основном confirmation участке средняя нормированная ошибка прогноза:

| Selector | MSE ↓ |
|---|---:|
| **MG** | **0.569** |
| Spectral entropy | 0.628 |
| Normalized increments | 0.656 |
| Fixed periodic | 0.661 |
| Fixed AR | 0.698 |
| Scalar recurrence | 0.528 |
| Oracle | 0.499 |

MG выбирает более подходящую модель, чем фиксированная periodic/AR стратегия и entropy. Однако recurrence лучше MG.

На двух более поздних участках trajectories MG сохраняет результат:

| Selector | MSE ↓ |
|---|---:|
| **MG** | **0.440** |
| Spectral entropy | 0.519 |
| Normalized increments | 0.501 |
| Fixed periodic | 0.551 |
| Fixed AR | 0.544 |
| Scalar recurrence | 0.356 |
| Oracle | 0.338 |

Без chaotic control средняя ошибка MG равна 0.453, recurrence — 0.405. Таким образом, MG даёт устойчивое преимущество над entropy и фиксированными моделями, но не над scalar recurrence.

## Что это показывает

Это практический use case: один scalar-log анализ выбирает downstream-модель и уменьшает ошибку прогноза без доступа к будущему участку. Преимущество MG связано с геометрической информацией о рекуррентной динамике, но recurrence остаётся сильным и более дешёвым конкурентом.

Эксперимент пока не добавлен в статью: для paper-level claim нужны новые независимо обученные генераторы и заранее зафиксированное сравнение с recurrence. Все исходные traces, frozen rules, transfer results и скрипты сохранены в `research_mg_forecast/`.
