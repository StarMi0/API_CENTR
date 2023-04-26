# Принцип работы:

1. На входе подаем архив из 8 файлов (8 бинарных масок формат jpg /png одноканальные GrayScale размер 1008 х 1344 пикселей и переменную mirr, которая может иметь 2 значения - mirr1 и mirr2
- Если mirr1 - отправлем на певую модель нейросети (g_model)
- Если mirr2 - отправлен на вторую модель нейросети (**необходимо будет прописать переменную для загрузки второй модели**)
2. На выходе архив из 8 файлов png одноканальные GrayScale размер 1008 х 1344 пикселей

# Произведенные изменения:
1. Добавлена документация, где все это отражено README.md
2. Добавлен файл [req.txt](./req.txt) в котором назначен список всех необходимых библиотек
1. Файлы [input data.py](./functions/input_data.py) [function.py](./functions/function.py) перенесены в папку [functions](./functions) и запускаются оттуда
3. [api.py](./api.py):
- Внесены изменения в импорт в связи с обновлением древа файлов.
- Убраны функции отзеркаливания изображений.
- Убраны функции переименовывания изображений.
- Добавлена переменная model_name1 = 'M_good190.h5' для выбора модели сетки, когда будет вторая, необходимо будет добавить название второй модели.
- Добавлена функция выбора модели для обработки изображения:
```python
if req_mirr == 'mirr1':
    imgs = predict_img(get_img_for_predict(unzipped), model_name1)
elif req_mirr == 'mirr2':
    imgs = predict_img(get_img_for_predict(unzipped), model_name1)
```
- Добавлен вывод стандартного архива для mirr2:
```python
# Временная заглушка на архив с изображениями
if req_mirr == 'mirr2':
    zip_file = 'satndart_zip.zip'
```
- Вернул стандартную функцию переименовывания:
```python
# сохраняем оригинальные имена файлов, создаем список новых имен файлов
new_file_names = ["0", "035", "090", "145", "180", "215", "270", "325"]

# переименовываем файлы
file_counter = 0
for filename in files:
    if filename.endswith('.png'):
        try:
            os.rename(os.path.join(unzipped, filename), os.path.join(unzipped, new_file_names[file_counter] + ".png"))
            file_counter += 1
        except:
            pass
```
4. [function.py](./functions/function.py):
- Все функции откомментированы.
- Изменена функция predict_img, в которой добавлена возможность выбора пути загрузки файла модели
```python
def predict_img(img, model):
    g_model = input_data.model_load(input_data.current_path, model)
    img = g_model.predict(img)[0] * 127.5 + 127.5
    return np.where(img < 128, 0, 255)  # shape np.array (2048, 2048, 8)
```
- Добавлена функция обрезки изображения до стандартизированного значения:
```python
def insert_array(original_array, insert_array):
    original_shape = np.shape(original_array)
    insert_shape = np.shape(insert_array)
    row_diff = int((original_shape[0] - insert_shape[0]) / 2)
    col_diff = int((original_shape[1] - insert_shape[1]) / 2)
    output_array = np.zeros(original_shape, dtype=int)
    output_array[row_diff:(row_diff + insert_shape[0]), col_diff:(col_diff + insert_shape[1])] = insert_array

    return output_array
```
5. [input_data.py](./functions/input_data.py):
- Добавлена функция model_load
```python
def model_load(path: str, model: str):    
    g_model = load_model(os.path.join(path, 'data', 'model', model), compile=True)
    return g_model
```
которая принимает на вход путь к файлу модели и название файла модели, сделано для реализации загрузки двух версий сетки