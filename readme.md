# РГР, Вальчевський П., ОІ-21сп
## Було обрану таку вибірку з Kaggle
https://www.kaggle.com/code/abdelrhmanyoussef/uae-realstate/input
Цей датасет містить інформацію про оренду нерухомості в ОАЕ + адресу, кількість спалень, ванних кімнат, тип нерухомості, площу, категорію оренди, ціну за квадратний метр та інші ознаки. Це відноситься до задачі регресії, оскільки можна прогнозувати орендну плату на основі різних змінних.
## Бізнес-проблема:
Задача, що вирішуєшться за допомогою цього датасету, полягає в прогнозуванні вартості оренди нерухомості (Rent) на основі різних характеристик, таких як: Beds  Baths Type  Area_in_sqft  Rent_per_sqft Rent_category Furnishing  Age_of_listing_in_days  City  Latitude  Longitude
## Опис проекту
Цей проект не включає підготовку даних, бо усе було зроблено у файлі rgr.ipynb, але є розділення на тренувальний та тестовий набори даних. Це потрібно для тренування та подальшого передбачення моделі.
Я вибрав таку модель `RandomForestRegressor` та здійснив їх налаштування (вибрав гіперпараметри та здійснив їх оптимізацію за допомогою `RandomizedSearchCV`).

## Структура проекту
- `.venv` - папка віртуального середовища проекту, де зберігаються ізольовані бібліотеки та залежності (автоматично створюється у середовищі). 
- `src/` - папка що містить усі виконавчі скрипти (або .ipynb, .pkl).
  - `split_data.py` - розподіл даних на тренувальний і тестовий набори та збереження у окремі файли у каталозі `data` (є однойменною функцією).
  - `train_models.py` - налаштування і тренування моделі на тренувальних даних (є однойменною функцією).
  - `predict_models.py` - передбачення на основі тестових наборів даних.
  - `random_forest_model.pkl` - збережена модель.
  - `rgr.ipynb` - файл звіт (для аналізу даних і виконання пунктів РГР)
- `data/` = папка, що містить набори даних (оригінальний, оброблений, тренувальний, тестовий).
  - `dubai_properties_processed_encoded.cvs` - оброблені та закодовані дані.
  - `train_split.cvs` - тренувальний набір даних.
  - `new_input.cvs` - тестовий набір даних.
- `README.md` - документація проекту.
- `requirements.txt` - перелік залежностей для відтворення середовища (згенеровано через `Tools -> Sync Python Requirements`).
- `main.py` - файл запуску програми, він реалізує послідовне виконання файлів `split_data.py, train_models.py, predict_models.py`.

## Установка та інструкція користувачеві
1. Клонування репозиторію:
```bash
   git clone https://github.com/ManFromLviv/basics_of_smart_technologies_and_systems_pr_1_to5
   ```
2. Перехід до папки проекту:
```bash
   cd <назви папки, де скопійовано проект>
```
3. Створення віртуально середовища:
```bash
   python -m venv .venv
```
4. Активування віртуального середовища:
```bash
   .venv\Scripts\activate
```
5. Встановлення залежностей:
```bash
   pip install -r requirements.txt
```
6. Запуск програми:
```bash
   python main.py
```