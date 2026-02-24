# Title

<!-- Change `kisnikser/m1p-template` to `intsystems/your-repository`-->
[![License](https://badgen.net/github/license/kisnikser/m1p-template?color=green)](https://github.com/kisnikser/m1p-template/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/kisnikser/m1p-template)](https://github.com/kisnikser/m1p-template/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/kisnikser/m1p-template.svg?color=0088ff)](https://github.com/kisnikser/m1p-template/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/kisnikser/m1p-template.svg?color=7f29d6)](https://github.com/kisnikser/m1p-template/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Nickolay Karlov </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Alexey Kravatskiy </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Vadim Strijov </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

Обучение глубоких моделей сопровождается генерацией множества взаимосвязанных временных рядов (loss, accuracy, weight parameters, learning rate). В данной работе предлагается новый метод анализа этих логов. Для выявления причинно-следственных связей в динамике обучения применяются Convergent Cross Mapping (CCM) и его модификации с опорой на теоремы Старка о delay embeddings для систем с детерминированным и стохастическим воздействиями. Исследуется нелинейные взаимосвязи данных рядов, а также предложен фреймворк, способный: анализировать воздействие изменения параметров на loss, детектировать Edge of Stability и осуществлять раннее предсказание эффекта grokking через отслеживание коллапса эффективной размерности вложения.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
