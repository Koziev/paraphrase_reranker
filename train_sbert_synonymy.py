"""
Модель детектирования парафраз на архитектуре sentence transformer.
Используется датасет с примерами правильных и неправильных перефразировок.
"""

import os
import glob
import json
import random

import tqdm

from sklearn.model_selection import train_test_split
import sentence_transformers
from sentence_transformers import SentenceTransformer, models, evaluation
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses


proj_dir = os.path.expanduser('~/polygon/chatbot')
tmp_dir = os.path.join(proj_dir, 'tmp')

batch_size = 2048
num_epochs = 30

dataset_path = os.path.join(proj_dir, 'tmp', 'paraphrase_detection_dataset.json')
print('Loading samples from "{}"...'.format(dataset_path))
all_examples = []
for sample in json.load(open(dataset_path, 'r')):
    # попарные комбинации текстов из раздела "paraphrases" дают позитивные сэмплы
    for i, phrase1 in enumerate(sample['paraphrases']):
        for phrase2 in sample['paraphrases'][i+1:]:
          all_examples.append(InputExample(texts=[phrase1, phrase2], label=1))

        # делаем попарные комбинации данных с примерами неверных перефразировок, чтобы получить негативные примеры
        if len(sample['distractors']) > 0:
            for phrase2 in random.choices(population=sample['distractors'], k=2):
                all_examples.append(InputExample(texts=[phrase1, phrase2], label=0))

train_examples, test_examples = train_test_split(all_examples, test_size=0.1)
print('Using {} samples'.format(len(train_examples)))

word_embedding_model = models.Transformer('cointegrated/rubert-tiny2')

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_loss = losses.ContrastiveLoss(model=model)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

output_dir = os.path.join(proj_dir, 'tmp', 'sbert_synonymy')

#test_dataloader = DataLoader(test_examples, shuffle=True, batch_size=batch_size)
evaluator = sentence_transformers.evaluation.BinaryClassificationEvaluator(sentences1=[s.texts[0] for s in test_examples],
                                                                           sentences2=[s.texts[1] for s in test_examples],
                                                                           labels=[s.label for s in test_examples],
                                                                           batch_size=256,
                                                                           show_progress_bar=False)
# удалим результаты оценки из предыдущих сессий
for p in glob.glob(os.path.join(output_dir, 'eval', '*.csv')):
    os.remove(p)

print('Start training sentence transformer for paraphrase detection...')
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          evaluator=evaluator,
          evaluation_steps=200,
          warmup_steps=warmup_steps,
          weight_decay=1e-5,
          output_path=output_dir,
          save_best_model=True,
          show_progress_bar=True)

print('All done. Look at the model saved in "{}" ;)'.format(output_dir))
