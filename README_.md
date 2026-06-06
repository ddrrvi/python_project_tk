# Поиск аномальных респондентов в SoS — Полусеместровый контроль №4

## Команда запуска

```bash
python solution_Дугаржапова_Дарима_ШЦТ_111.py
```

По умолчанию скрипт ищет `.parquet`-файлы в текущей директории и создаёт папку `output/`.

**С явным указанием пути к данным и выходной директории:**
```bash
python solution_Дугаржапова_Дарима_ШЦТ_111.py --data_dir ./data_train --output_dir ./output
```

---

## Структура выходных файлов

```
output/
├── anomalies.csv              # SubjectID, researchdate — пары для удаления
├── anomaly_reasons.csv        # диагностика: бренд, score, threshold, reason
└── plots/
    ├── total_ots_before_after.png   # OTS до/после по дням
    ├── category_ots_change.png      # Изменение OTS по категориям (%)
    └── daily_anomaly_count.png      # Число аномальных респондентов по дням
```

---

## Описание алгоритма

### Что считается аномалией

Аномалия — это **чрезмерно высокая** дневная активность респондента по конкретному бренду.  
Малый OTS **не является** аномалией; все правила направлены только на выбросы сверху.

### Как считается score и пороги

Для каждой тройки `(SubjectID, researchdate, BrandID)` вычисляется:

```
daily_ots(i, j, k) = Weight(i, k) * count_rows(i, j, k)
```

Затем внутри каждой группы `(researchdate, BrandID)` применяются два критерия:

#### Критерий A — Robust Z-Score

```
rzscore = (daily_ots - median) / (1.4826 * MAD)
```

- Порог `RZSCORE_THRESHOLD = 3.5` соответствует хвосту далеко за 99.9-м перцентилем нормального распределения.
- Если `MAD (медианное абсолютное отклонение) = 0` (все значения одинаковы) устойчив к выбросам в отличие от стандартного отклонения., fallback на IQR-нормировку.
- **Защита от малых выборок**: для групп с `< 5` наблюдениями robust z-score не применяется.
- 1.4826 нормирует MAD для нормального распределения (≈ σ).
- Для групп < MIN_GROUP_SIZE этот критерий не применяется.

#### Критерий B — Доля в OTS бренда за день

```
share = daily_ots(i, j, k) / sum(daily_ots(*, j, k))
```

Порог `SHARE_THRESHOLD = 0.50`: если один респондент даёт ≥ 50% суммарного OTS бренда за день — это аномальная концентрация, экономически несовместимая с репрезентативной панелью.

Оба критерия **статистически обоснованы** и не содержат hardcode конкретных ID, дат или брендов.
#### Критерий C — Абсолютный перцентильный порог (внутри бренд-дня):
    Порог: daily_ots > Q75 + IQR_FACTOR × IQR  (Tukey fence)
    IQR_FACTOR = 3.0 (расширенный, чтобы не удалять норму)
    • Классический метод обнаружения выбросов Тьюки.
    • Направлен только вверх (нет нижней границы).
    • Применяется при группе ≥ MIN_GROUP_SIZE.
### Единица удаления

Если респондент признан аномальным хотя бы по одному бренду в конкретный день — в `anomalies.csv` добавляется пара `(SubjectID, researchdate)`, и при пересчёте OTS удаляются **все** строки этого респондента за этот день.

---

## Аналитические возможности (п. 8.2)

Все функции доступны для импорта из скрипта:

```python
from solution_ФИО_ГРУППА import (
    plot_before_after_by_column,
    get_query_texts,
    plot_brand_ots_before_after,
)

# До/после по полу
plot_before_after_by_column(df, anomaly_pairs, col='Пол',
                             output_path='output/plots/before_after_gender.png')

# До/после по типу ресурса
plot_before_after_by_column(df, anomaly_pairs, col='ResourceType',
                             output_path='output/plots/before_after_resourcetype.png')

# До/после по платформе
plot_before_after_by_column(df, anomaly_pairs, col='Platform',
                             output_path='output/plots/before_after_platform.png')

# До/после по возрасту
plot_before_after_by_column(df, anomaly_pairs, col='Возраст',
                             output_path='output/plots/before_after_age.png')

# До/после по региону
plot_before_after_by_column(df, anomaly_pairs, col='Регион',
                             output_path='output/plots/before_after_region.png')

# До/после по категории
plot_before_after_by_column(df, anomaly_pairs, col='CategoryDelivery',
                             output_path='output/plots/before_after_category.png')

# Поисковые запросы конкретного аномального респондента за день
df_queries = get_query_texts(df, subject_id=1585271561336880000, researchdate='2025-05-07')
print(df_queries[['QueryText', 'Brand', 'CategoryDelivery', 'ResourceName']].to_string())

# OTS по конкретному бренду до/после
plot_brand_ots_before_after(df, anomaly_pairs, brand_name='redmi',
                             output_path='output/plots/brand_redmi.png')
```

---

## Зависимости

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
pyarrow>=10.0.0   # для чтения .parquet
scipy>=1.9.0      # опционально
```

Установка: `pip install -r requirements.txt`
