from pathlib import Path
import json,hashlib,zipfile,subprocess,sys,shutil,re
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parent
def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def main():
 full=pd.read_csv(H/'final_audited_summary.csv').set_index('method');query=pd.read_csv(H/'query_fresh_summary.csv').set_index('method');tim=pd.read_csv(H/'query_paired_summary.csv');aud=json.loads((H/'query_audit.json').read_text())
 assert aud['same_decisions']==aud['n_forecasts']==160
 primary=pd.read_csv(H/'query_fresh_comparisons.csv');paired=primary.groupby(['seed','method']).error.mean().unstack()
 gain=1-query.loc['MG_q128','mean']/query.loc['validation','mean']
 models=['MG_q128','MG','validation','TwoNN','LB','PRdelay','permutation_entropy','sample_entropy','cheap','cheap_val','val_MG','all_nonMG','oracle']
 names={'MG_q128':'MG, 128 опорных точек','MG':'Полный MG','validation':'Обычный validation-перебор','TwoNN':'TwoNN','LB':'Levina--Bickel','PRdelay':'Ранг ковариации delay-облака','permutation_entropy':'Permutation entropy','sample_entropy':'Sample entropy','cheap':'Дешёвые признаки','cheap_val':'Дешёвые признаки + validation','val_MG':'Validation errors + MG','all_nonMG':'Все признаки без MG','oracle':'Oracle с будущими ответами'}
 results=table(['Правило выбора','Ошибка, меньше лучше'],[[names[k],f'{query.loc[k,"mean"]:.5f}'] for k in models])
 compact=pd.read_csv(R/'research_mg_ucr/screen_summary.csv')
 fig,axs=plt.subplots(1,2,figsize=(9,3.0),layout='constrained')
 for method,label in [('MG_q128','MG: 128 queries'),('validation','Validation'),('cheap_val','Cheap + validation')]:axs[0].plot(paired.index,paired[method],'.-',label=label,lw=1)
 axs[0].set(xlabel='New generator seed',ylabel='Normalized forecast MSE',title='All 20 seed groups');axs[0].legend(fontsize=7)
 vals=[tim.MG.median(),tim.MG_q128.median(),tim.validation.median()];axs[1].bar(['Full MG','128 queries','Validation'],np.array(vals)*1000);axs[1].set(ylabel='Milliseconds',title='Same-input end-to-end timing')
 for ax in axs:ax.grid(axis='y',alpha=.2)
 fig.savefig(H/'campaign_result.pdf');fig.savefig(H/'campaign_result.png',dpi=170);plt.close(fig)
 text=r'''# Где MG приносит практическую пользу: итог расширенного поиска

## Главный результат

Найден применимый вариант: выбирать модель короткого прогноза активности обученной рекуррентной сети по MG одного нейрона. Полный MG показывает хорошую среднюю точность, но дорог. Приближённый MG по 128 опорным точкам сохранил все решения полного метода на отдельной новой серии и ускорил этот выбор примерно в 16 раз. При этом нельзя утверждать, что MG статистически превосходит всех конкурентов: у более сложных обученных правил средняя ошибка бывает меньше.

## Что обучается и зачем

Сначала FORCE/RLS обучает рекуррентную сеть с 256 нейронами генерировать заданные колебания. Есть семь заданий: одна--четыре независимые частоты, два/четыре гармонических выхода и две независимые частоты с гармониками. Восьмая группа — необученная сеть с большим recurrent gain; она может прийти и к постоянному сигналу, поэтому название chaos не означает доказанного хаоса. Все группы сохраняются, даже если сеть плохо воспроизводит цель.

После обучения веса фиксируются. Доступна только активность третьего нейрона. Нужно прогнозировать следующие 32 отсчёта по предыдущим 2048. Прикладное действие — выбрать прогнозную модель, чтобы получить меньшую ошибку без перебора всех моделей на каждом новом сигнале. Это контролируемая задача прогнозирования нейронной активности, а не обучение LLM, ранняя остановка PPO или реальная робототехника.

Восемь кандидатов: последнее значение, среднее, повтор последнего приблизительного цикла, Ridge с 8/48 лагами, 5 ближайших соседей с 8/48 лагами и Ridge на 64 случайных Fourier-признаках с 48 лагами. Все прогнозы прямые многомерные, с одинаковой нормировкой по известному прошлому. Модели обучаются только по префиксу. Ошибка — MSE будущего блока, делённая на дисперсию префикса; это не RMSE.

MG вычисляется при E=20, адаптивном лаге, 20 соседях и Theiler exclusion с пределом 150. Небольшое дерево учится предсказывать ошибки кандидатов по MG и выбирает наименьшую. На простых циклах оно обычно выбирает локальный прогноз ближайшими соседями; при более сложной динамике — Ridge. Это эмпирическое правило, не теорема о единственно правильной прогнозной модели.

## Как разделены поиск и проверка

Исходные пять больших генераторов по каждой задаче уже использовались в прошлой работе. Они послужили только для поиска. Затем сохранено 100 новых seed-групп 601--700: в каждой семь обучений по 40000 шагов и одна необученная контрольная сеть. До этого запроса группы 601--618 уже существовали; в данном продолжении проведено 574 новых обучения и добавлено 82 контрольных сети.

Seeds601--618 обучают правила выбора. Seeds619--630 выявили перспективный короткий горизонт. На следующей серии631--650 правила уже фиксированы; затем добавлены более сильные геометрические конкуренты. Seeds651--680 являются отдельной проверкой полного набора сравнений. На ней полный MG получил ошибку 0.03139 против 0.03224 у validation и 0.03540 у TwoNN. В разнице с validation интервал включает ноль. Результат на предыдущих20сидах был заметно сильнее, поэтому его нельзя выдавать за окончательную оценку эффекта.

После замера высокой стоимости полного MG предложено фиксировать 128 равномерно распределённых опорных delay-векторов. Всё облако сохраняется как кандидаты в соседи; временное исключение не меняется. Два других бюджета64/256 — проверки чувствительности, основной128 выбран заранее. Эту новую аппроксимацию проверяют отдельные seeds681--700, без изменения уже обученного дерева.

## Итог на последних 20 seed-группах

Серия содержит160 прогнозных задач, но независимая единица статистики —20 seed-групп, а не160 записей. Все восемь режимов учтены. На постоянных сигналах все алгоритмы используют одно правило: последний отсчёт; NaN не превращается в нулевую размерность.

RESULTS_TABLE

У 128-query MG ошибка примерно на10.7% ниже обычного validation-перебора, но95%-й bootstrap-интервал парной разницы равен [-0.00729;0.01758] и включает ноль. В9 из20групп MG лучше validation; преимущества по каждому сиду нет. Наиболее точное из показанных обученных правил использует validation errors вместе с MG и имеет ошибку0.03167. Дешёвые признаки+validation без MG дают0.03379, также лучше отдельного MG. Поэтому утверждение «лучше всех аналогов» данными не подтверждается.

Зато все160 решений128-query версии совпали с полным MG, и все ошибки совпали. В этой серии результаты64/256query также совпали; это наблюдение на данной семье, не гарантия для других данных. Доверительные интервалы и индивидуальные seed находятся в query_intervals.csv и query_fresh_comparisons.csv.

![Качество по всем новым сидам и стоимость](campaign_result.pdf)

## Вычислительный выигрыш

TIMING_TABLE

Здесь время включает вычисление признака, выбор и обучение выбранного прогнозатора. Validation четыре раза проверяет каждый из восьми кандидатов на известных последних128точках и затем обучает выбранного на всём префиксе:33 fit/predict операции. В замере нет загрузки файлов, обучения исходных нейросетей и обучения дерева выбора. Стоимость первого обучения дерева и сбора его calibration-ошибок должна амортизироваться на множестве последующих прогнозов.

Замеры выполнялись последовательно, с одним численным потоком, на одних входах: seeds681,690,700, четыре задачи в каждом, семь повторов, перемешанный порядок методов. Это12 входных рядов, а не160 независимых замеров времени. Полный MG здесь медленнее дешёвых признаков; ускорение относится к новой аппроксимации. Скорость зависит от реализации и оборудования.

Аппроксимация использует тот же estimator kernel, но вместо n запросов делает q запросов и возвращает (q(k-1)-1)/сумма локальных логарифмов расстояний. Проверено точное совпадение с исходным MG при q=n, совпадение расстояний KDTree и блочного поиска, affine invariance, отклонение констант и нечисловых значений. Это новое приближение; название «исходный MG без изменений» было бы неверным.

## Что ещё проверено и почему не объявлено успехом

Доработанное сравнение гармонической модели с несколькими независимыми частотами проиграло обычной валидации: ошибки0.2084 у MG,0.1640 у holdout и0.1909 у фиксированной многокчастотной модели. Перед запуском исправлены общая нормировка прошлого/будущего и поиск раздельных спектральных пиков; это необходимые исправления, не подбор положительного результата.

В новой серии классификации проверены16 официальных UCR-наборов. Сравнивались20 временных/спектральных признаков,8 participation ratio,8 MG, одинаковые SVC/ExtraTrees и raw-waveform модели. Классификатор выбирался только по training CV. Для перспективных задач добавлены DTW и независимая реализация1000 случайных convolution kernels (ROCKET-style), но это не полный оригинальный benchmark ROCKET.

На TwoLeadECG добавление MG дало91.31% balanced accuracy против82.19% у того же классификатора без MG. Однако ROCKET-style достиг99.91%: компактная модель улучшилась, общего превосходства нет. На ToeSegmentation1 добавление MG дало87.78% против83.61%, но ROCKET-style получил96.11%. На Earthquakes MG-признаки дали64.59% против61.65% в matched ablation; этот небольшой тест139записей остаётся exploratory, а поправка на16исследованных наборов не поддерживает сильный вывод. Классификация ЭКГ здесь не является медицинской рекомендацией.

На Wafer сначала получился почти идеальный результат даже при20размеченных примерах. Аудит показал, что нормальные сигналы имеют много повторяющихся значений, а модель использует отсутствие оценки MG. Три простых признака повторов дают100% balanced accuracy. Поэтому это найденный артефакт dataset/измерения, а не достоинство MG. На ItalyPowerDemand длины24недостаточно для большинства MG-конфигураций; отсутствующие оценки сохранены, а benchmark не годится для выводов о размерности. В ECG5000 одна training-категория содержит2примера, что ограничивает3-foldCV.

Предыдущие UCI traffic/energy,12сенсорных forecasting-сценариев, HAR и certification stopping не превзошли сильные baseline. Старые файлы сохранены; сценарии не исчезают из отчёта при смене гипотезы. Выигрыш нейросетевого forecast routing не означает доказанного уменьшения active dimension и не доказывает, когда завершать обучение.

## Литература и дальнейшее использование

FFORMA мотивирует выбор и усреднение прогнозных моделей по признакам ряда: https://www.monash.edu/business/ebs/research/publications/ebs/wp19-2018.pdf . Catch22 даёт сильный компактный контекст для временных признаков: https://arxiv.org/abs/1901.10200 . Попытка установить pycatch22 не удалась из-за отсутствия MSVC; наши handcrafted20 не называются catch22. Данные: UCR/UEA https://www.timeseriesclassification.com/ , все downloadURL и SHA256 сохранены в source.json. Оценивание следовало официальным train/testsplit; там, где ID людей/станков отсутствуют, независимость по ним не доказана.

Практическая формулировка результата: «В контролируемом прогнозировании активности обученных рекуррентных сетей MG выбирает прогнозную модель по одному сигналу. Аппроксимация по128опорным точкам на новой серии сохранила160/160решений полного метода и ускорила маршрутизацию примерно в16раз; по средней ошибке она сопоставима с более дорогим validation-перебором». Для утверждения о преимуществе над всеми методами данных недостаточно. Эта серия пока не добавлялась в статью.
'''
 trows=[['Полный MG',f'{tim.MG.median()*1000:.2f}'],['MG:128queries',f'{tim.MG_q128.median()*1000:.2f}'],['Validation-перебор',f'{tim.validation.median()*1000:.2f}']]
 text=text.replace('RESULTS_TABLE',results).replace('TIMING_TABLE',table(['Метод','Медиана времени, мс'],trows))
 text=re.sub(r'(?<=[а-яА-Я])(?=\d)|(?<=\d)(?=[а-яА-Я])',' ',text)
 text=text.replace('многокчастотной','многочастотной').replace('Seeds601','Seeds 601').replace('Seeds619','Seeds 619').replace('Seeds651','Seeds 651').replace('seeds681','seeds 681')
 (H/'report_ru.md').write_text(text,encoding='utf-8')
 build=(R/'research_walker_smooth/build_report.py').read_text(encoding='utf-8')
 # Render plain URLs as breakable clickable links; escape monospace paths in normal prose normally.
 build=build.replace('\\usepackage[hidelinks]{hyperref}',r'\usepackage{xurl}'+'\n'+r'\usepackage[hidelinks]{hyperref}')
 build=build.replace(r'\setmainfont{Times New Roman}',r'\setmainfont{Times New Roman}'+'\n'+r'\setmonofont{Courier New}'+'\n'+r'\newfontfamily\cyrillicfonttt{Courier New}')
 build=build.replace("else:body.append(inline(s))", "else:\n        parts=re.split(r'(https?://[^ ]+)',s)\n        body.append(''.join(r'\\url{'+p+'}' if p.startswith('http') else inline(p) for p in parts))")
 (H/'build_report.py').write_text(build,encoding='utf-8')
 subprocess.run([sys.executable,str(H/'build_report.py')],check=True)
 # Code/results bundle; external corpora and hundreds of MB of per-neuron traces remain local.
 archive=R/'mg_search_results.zip';files=[]
 for root in [H,R/'research_mg_ucr']:
  for p in root.rglob('*'):
   if not p.is_file() or '__pycache__' in p.parts:continue
   if p.suffix not in {'.py','.md','.csv','.json','.pkl','.pdf'}:continue
   if p.name.startswith('features_') and root.name=='research_mg_ucr':continue
   files.append(p)
 files +=[R/'research_mg_forecast/model_order.py',R/'research_mg_forecast/MODEL_ORDER_PROTOCOL.md',R/'research_mg_forecast/model_order_summary.csv',R/'research_mg_forecast/model_order_records.csv']
 manifest=[dict(path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
 (H/'search_archive_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in files:z.write(p,p.relative_to(R).as_posix())
  z.write(H/'search_archive_manifest.json','search_archive_manifest.json')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for entry in manifest:assert hashlib.sha256(z.read(entry['path'])).hexdigest()==entry['sha256']
 print('Search archive:',archive.stat().st_size,'bytes; hashes verified')
if __name__=='__main__':main()
