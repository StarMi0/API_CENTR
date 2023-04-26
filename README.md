# Принцип работы:

1. На входе подаем архив из 8 файлов (8 бинарных масок формат jpg /png одноканальные GrayScale размер 1008 х 1344 пикселей и переменную mirr, которая может иметь 2 значения - mirr1 и mirr2
- Если mirr1 - отправлем на певую модель нейросети (g_model)
- Если mirr2 - отправлен на вторую модель нейросети (**необходимо будет прописать переменную для загрузки второй модели**)
2. На выходе архив из 8 файлов png одноканальные GrayScale размер 1008 х 1344 пикселей

# Произведенные изменения:
1. Добавлена документация, где все это отражено README.md
1. Файлы [input data.py](./functions/input_data.py) [function.py](./functions/function.py) перенесены в папку [functions](./functions) и запускаются оттуда
3. [api.py](./api.py):
- Внесены изменения в импорт в связи с обновлением древа файлов
- Убраны функции отзеркаливания изображений
- Добавлена функция выбора модели для обработки изображения:
```python
if req_mirr == 'mirr1':
    imgs = predict_img(get_img_for_predict(unzipped), model_name1)
elif req_mirr == 'mirr2':
    imgs = predict_img(get_img_for_predict(unzipped), model_name1)
```
4. [function.py](./functions/function.py):
- Изменена функция predict_img, в которой добавлена возможность выбора пути загрузки файла модели
```python
def predict_img(img, model):
    g_model = input_data.model_load(input_data.current_path, model)
    img = g_model.predict(img)[0] * 127.5 + 127.5
    return np.where(img < 128, 0, 255)  # shape np.array (2048, 2048, 8)
```
5. [input_data.py](./functions/input_data.py):
- Добавлена функция model_load
```python
def model_load(path: str, model: str):    
    g_model = load_model(os.path.join(path, 'data', 'model', model), compile=True)
    return g_model
```
которая принимает на вход путь к файлу модели и название файла модели, сделано для реализации загрузки двух версий сетки