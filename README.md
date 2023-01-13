# Модель ранжирования перефразировок

Модель с архитектурой [sentence transformer](https://www.sbert.net/) для определения близости семантики двух **коротких** фраз.
Разработана и поддерживается для проекта [диалоговой системы](https://github.com/Koziev/chatbot).

На вход модели подается список предложений. Ны выходе получается набор векторов эмбеддингов этих предложений. Косинус
между парой векторой дает оценку близости предложений по смыслу. Таким образом, примерное равенство:

```
Одна голова - хорошо, а две - лучше  ≅  пара голов имеет преимущество перед единственной
```

находит отражение в близости косинуса соответствующих векторов к 1.

А для несинонимичных фраз:

```
Одна голова - хорошо, а две - лучше  ≇ Потерявши голову, по волосам не плачут
```

косинус между их векторами будет близок к 0.

Выполнить соответствующие оценки можно таким кодом:

```
import sentence_transformers

sentences = ["Одна голова - хорошо, а две - лучше",
"пара голов имеет преимущество перед единственной",
"Потерявши голову, по волосам не плачут",]

model = sentence_transformers.SentenceTransformer('inkoziev/sbert_synonymy')
embeddings = model.encode(sentences)

s1 = sentences[0]
v1 = embeddings[0]
for i2 in range(1, 3):
    s = sentence_transformers.util.cos_sim(a=v1, b=embeddings[i2]).item()
    print('text1={} text2={} cossim={}'.format(s1, sentences[i2], s))
```

Результат будет примерно таким:

```
text1=Одна голова - хорошо, а две - лучше  text2=пара голов имеет преимущество перед единственной  cossim=0.8603419065475464
text1=Одна голова - хорошо, а две - лучше  text2=Потерявши голову, по волосам не плачут  cossim=0.013120125979185104
```


## Обучение

Публичная версия датасета, на которых обучалась модель, [доступна на huggingface](https://huggingface.co/datasets/inkoziev/paraphrases). В этом датасете
есть примеры неправильных перефразировок, которые используются в качестве негативных примеров при обучении данной модели.

Тренировка выполняется кодом [train_sbert_synonymy.py](train_sbert_synonymy.py).

## Использование

Готовая модель и описание способа ее использования [находятся на huggingface](https://huggingface.co/inkoziev/sbert_synonymy).

## Сопряженные проекты

Генеративный поэтический перефразировщик доступен в хабе hugginggace: [inkoziev/paraphraser](https://huggingface.co/inkoziev/paraphraser).
Код обучения этой модели доступен в [репозитории на гитхабе](https://github.com/Koziev/paraphraser).



