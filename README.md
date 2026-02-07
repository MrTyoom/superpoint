#### **Подготовка и активация окружения**
```bash
    # setup uv environment 
    # + install requirements 
    # + pre-commit hook
    make setup

    # активация окружения
    source .venv/bin/activate
```

#### **DVC pipeline**

```bash
    # генерация синтетических данных
    dvc repro prepare_synthetic_data
```

#### **Pre-commit check**

```bash
    make pre-commit-check
```
