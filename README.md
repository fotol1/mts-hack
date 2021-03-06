# Решение кейса "Умные покупки от МТС" команды TopPopular

### Краткое описание кейса

В качестве задачи предлагалось построить рекомендательную систему кэшбека для клиентов МТС Банка. Были даны данные о транзакционной активности клиентов, а также информация по соцдему. У транзакций были известны MCC коды, а также "грязная" строка с названием и адресом точки продаж.

Рекомендательная система должна предлагать персональные офферы клиенту, исходя из его интересов.

### Краткое описание нашего решения

За время хакатона мы успели построить двухуровневую модель рекомендаций для MCC/merchant_nm. В качестве базовых моделей для первого уровня были взяты стандартные бейзлайны EASE, SLIM, MultiVAE. Дальше их предсказания объединялись с соцдем признаками клиентов, и статистиками по MCC/магазинам. Объединенные данные обрабатывал градиентный бустинг (модель второго уровня).

Данные разделили по каждому пользователю в соотношении 80%/10%/10% для обучения/валидации/теста соответственно. Так мы старались приблизить сценарий оценивания к реальному использованию моделей.

Градиентный бустинг обучен на интеракциях валидационной выборки. Конечные метрики подсчитаны по тесту. Эти интеракции не видели ни модели первого уровня, ни градиентный бустинг. Поэтому считаем, что конечное сравнение моделей между собой корректно.

Таким образом на выходе мы получили модель, которая учитывает всю доступную информацию по пользователю, динамически подсчитывает характеристики магазина/mcc-категории. Также модель сильно зависит от того, где клиент совершает транзакции (это следует из того, что самый важный признак принадлежит модели).

### Как запустить решение

Необходимо в положить в папку Data датасеты `train_1.csv`, `train_2.csv`.

Для более легкой воспроизводмости модель можно воспроизвести через docker. Необходимо собрать образ через Dockerfile и запустить контейнер. Для удобства можно выполнить

```
chmod +x docker.sh
./docker.sh
```
Теперь у нас есть необходимые зависимости, и мы можем запускать код. Сначала подготовим данные в нужный формат. Потом обучим модели 

```
cd src
python3 process_data.py --item_col=mcc --nrows=1000000
python3 train_models.py --item_col=mcc
```

```
Теперь обучим для магазинов
python3 process_data.py --item_col=merchant --nrows=1500000
python3 train_models.py --item_col=merchant
```

После выполнения этого скрипта в папке `Data/artifacts` появятся файлы c весами моделей.

Теперь можно составить предсказания на тестовых пользователях. Для этого нужно запустить
```
python3 get_predictions.py --transaction_path=PATH1 
``` 
Предсказания лучшей MCC категории будут сложены сюда `Data/predictions.csv.gz`.

Так как нашая основная модель коллаборативной фильтрации MultiVAE умеет отдавать предсказания только на основе интеракций, мы также сделали элементарное API, которое позволяет получать предсказания по пользователю, на основе тех MCC-категорий, в которых он совершал покупки. Внутри докера можно запустить сервер
``` 
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
``` 
И получить рекомендации для пользователя, который, например, совершал транзакции в магазинах. 10 (litres) и 61 - внутрение айди магазинов 

`http://127.0.0.1:8000/merchant_recommendations/?q=10__18`

Для рекомендаций категорий:

`http://127.0.0.1:8000/category_recommendations/?q=5411__5897`





### Что мы не успели сделать в рамках хакатона


- Добавить модель на трансформере SASRec
- Подбирать гиперпарметры у моделей через валидацию
- Использовать гео


