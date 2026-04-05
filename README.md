#### **Подготовка и активация окружения**
```bash
    # setup uv environment 
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create a .venv local virtual environment (if it doesn't exist)
    [ -d ".venv" ] || uv venv
    
    # install requirements + pre-commit hook
    make setup

    # activate environment
    source .venv/bin/activate
```

#### **DVC pipeline**

```bash
    # генерация синтетических данных
    dvc repro prepare_synthetic_data
    
    # обучение модели MagicPoint на синтетических данных
    dvc repro train_magicpoint
    
    # скачивание спутниковых снимков с TMS сервисов
    dvc repro download_satellite_data
    
    # подготовка данных для экспорта ключевых точек
    dvc repro prepare_satellite_images
    
    # экспорт ключевых точек на спутниковые изображения
    dvc repro export_points
    
    
```

#### **Pre-commit check**

```bash
    make pre-commit-check
```
