# Модель определения и ранжирования перефразировок

Модель с архитектурой [sentence transformer](https://www.sbert.net/) для определения близости двух коротких фраз.
Разработана и поддерживается для проекта [диалоговой системы](https://github.com/Koziev/chatbot).

## Обучение

Публичная версия датасета, на которых обучалась модель, [доступна на huggingface](https://huggingface.co/datasets/inkoziev/paraphrases).

Тренировка выполняется кодом [train_sbert_synonymy.py](train_sbert_synonymy.py).


## Использование

Готовая модель и описание способа ее использования [находятся на huggingface](https://huggingface.co/inkoziev/sbert_synonymy).
