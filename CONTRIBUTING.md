# Contributing to gost-rag

[English](#english) · [Русский](#русский)

---

## English

Thanks for your interest in the project!

### How to propose changes

1. Fork the repository or create a branch from `main`.
2. Branch naming: `feature/short-description` or `fix/short-description`.
3. Write commits with clear messages (see below).
4. Open a Pull Request to `main`.
5. Describe what and why you change. If it closes an issue — add `Closes #N`.

### Commit style

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — refactoring without behavior change
- `test:` — tests
- `chore:` — routine (dependencies, configs)

Examples:
```
feat: add PDF loader
fix: correct chunk overlap in splitter
docs: update installation instructions
```

### Running locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Reporting issues

When opening an issue, please include:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Environment: OS, Python version, relevant library versions

---

## Русский

Спасибо за интерес к проекту!

### Как предложить изменения

1. Форкните репозиторий или создайте ветку от `main`.
2. Название ветки: `feature/краткое-описание` или `fix/краткое-описание`.
3. Делайте коммиты с понятными сообщениями (см. ниже).
4. Откройте Pull Request в `main`.
5. Опишите, что и зачем меняете. Если закрывает issue — укажите `Closes #N`.

### Стиль коммитов

Используем [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — новая функциональность
- `fix:` — исправление бага
- `docs:` — только документация
- `refactor:` — рефакторинг без изменения поведения
- `test:` — тесты
- `chore:` — рутина (зависимости, конфиги)

Примеры:
```
feat: add PDF loader
fix: correct chunk overlap in splitter
docs: update installation instructions
```

### Запуск локально

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Сообщения об ошибках

При открытии issue указывайте:
- Что ожидали получить
- Что получили на самом деле
- Шаги воспроизведения
- Окружение: ОС, версия Python, версии релевантных библиотек