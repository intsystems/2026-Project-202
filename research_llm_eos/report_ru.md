# MG при мониторинге обучения decoder-only Transformer

## Итог

Проведён локальный proof of concept на небольшом decoder-only Transformer, обучаемом предсказывать следующий символ в Tiny Shakespeare.

После десятикратного снижения learning rate в середине обучения MG по фиксированному validation probe loss снизился во всех трёх seed примерно на 30%. Это показывает, что MG чувствителен к упрощению наблюдаемой динамики после стабилизации обучения.

Строгую трактовку этого числа как абсолютной active dimension пока давать нельзя: identifiability ratio остаётся примерно 1.34–1.48, а validation loss является стохастическим и в основном транзиентным сигналом.

## Постановка

- decoder-only Transformer: 2 блока, width 64, 4 heads, context 64, около 108 тыс. параметров;
- задача: next-character prediction;
- корпус: Tiny Shakespeare;
- optimizer: AdamW;
- 3 seed;
- режимы: constant LR 0.003 и LR switch 0.003 → 0.0003 на шаге 1536.

Сохранялись training loss, fixed validation probe loss и gradient norm. На 21 checkpoint считались Hessian top eigenvalue, activation participation ratio и gradient-variance proxy.

## Основной результат

Для probe loss отношение MG после и до смены LR:

| Режим | Seed 0 | Seed 1 | Seed 2 | Медиана |
|---|---:|---:|---:|---:|
| Constant LR | 1.00 | 1.05 | 1.02 | 1.02 |
| LR switch | 0.70 | 0.71 | 0.67 | 0.70 |

При этом для LR switch медианный Hessian top eigenvalue изменился с 81.8 до 89.1, а activation PR — с 9.60 до 11.25. Поэтому падение MG не объясняется простым уменьшением curvature или effective rank активаций.

## Скорость

Медиана по трём AdamW seed и 21 checkpoint:

| Метод | Время |
|---|---:|
| Hessian top eigenvalue | 12.24 с |
| MG | 0.16 с |
| MG + validity checks | 0.37 с |
| Gradient proxy | 0.27 с |
| Activation PR | 0.06 с |

Поэтому MG примерно в 80 раз быстрее Hessian как postprocessing уже записанного scalar log. Если считать дополнительный forward для probe loss на каждом шаге, преимущество уменьшается примерно до 1.6 раза.

## Ограничения

Этот запуск пока нельзя подавать как доказательство теоретически гарантированной active dimension в LLM training. Лог не является явно рекуррентным, а diagnostics не дают надёжной абсолютной интерпретации.

Корректная формулировка результата: **MG дешёво обнаруживает смену динамического режима в scalar log обучения Transformer; в данном эксперименте он фиксирует стабилизацию probe-loss dynamics после снижения learning rate.**

Для публикационного результата нужен Colab GPU: модель хотя бы 10–50M параметров, 4–5 seed, более длинные runs и заранее зафиксированные окна и критерии валидности.
