import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

def main():

    # 🔧 КОНФИГУРАЦИЯ ДЛЯ МАЛЕНЬКОЙ МОДЕЛИ
    class FastModelConfig:
        # 🎯 МАЛЕНЬКИЕ И БЫСТРЫЕ МОДЕЛИ (выбери одну)
        model_options = {
            "rubert-tiny": "cointegrated/rubert-tiny2",  # ⚡ ОЧЕНЬ БЫСТРАЯ
            "distilrubert": "ai-forever/ruDistilBert",   # ⚡ БЫСТРАЯ
            "rubert-small": "sberbank-ai/ruBert-base",   # 🚀 БАЛАНС
            "multilingual": "bert-base-multilingual-uncased"  # 🌍 Мультиязычная
        }
        
        # Выбираем модель (поменяй на нужную)
        model_name = model_options["rubert-small"]
        dataset_name = "MonoHime/ru_sentiment_dataset"
        
        # НАСТРОЙКИ ДЛЯ СКОРОСТИ
        batch_size = 16             # Большой батч для маленькой модели
        max_length = 256            # Можно больше текста
        gradient_accumulation =4   # Маленький accumulation
        learning_rate = 3e-5        # Стандартный LR
        num_epochs = 2              # Быстрое обучение
        fp16 = True
        
        # 💾 Дополнительные настройки
        warmup_ratio = 0.05
        weight_decay = 0.01

    config = FastModelConfig()

    print(f"🎯 Выбрана модель: {config.model_name}")

    print("=== ЗАГРУЗКА ДАТАСЕТА ===")

    def load_dataset_fast():
        try:
            # Загружаем датасет
            dataset = load_dataset(config.dataset_name)
            print(f"✅ Датасет загружен: {config.dataset_name}")
            
            # 🔧 МОЖЕМ ПОЗВОЛИТЬ СЕБЕ БОЛЬШЕ ДАННЫХ
            max_train_samples = 20000  # 100к примеров!
            max_eval_samples = 2000    # 10к для валидации
            
            if len(dataset['train']) > max_train_samples:
                print(f"🔄 Ограничиваем train с {len(dataset['train'])} до {max_train_samples} примеров")
                dataset['train'] = dataset['train'].select(range(max_train_samples))
            
            # Создаем validation split если нужно
            if 'validation' not in dataset and 'valid' not in dataset and 'val' not in dataset:
                print("🔄 Создаем validation split...")
                train_valid_split = dataset['train'].train_test_split(
                    test_size=0.1,
                    seed=42
                )
                dataset = DatasetDict({
                    'train': train_valid_split['train'],
                    'validation': train_valid_split['test']
                })
                eval_split = 'validation'
            else:
                eval_split = 'validation' if 'validation' in dataset else 'valid'
                if len(dataset[eval_split]) > max_eval_samples:
                    print(f"🔄 Ограничиваем {eval_split} с {len(dataset[eval_split])} до {max_eval_samples} примеров")
                    dataset[eval_split] = dataset[eval_split].select(range(max_eval_samples))
            
            print(f"\n Финальные размеры:")
            print(f"   Train: {len(dataset['train'])} примеров")
            print(f"   Eval ({eval_split}): {len(dataset[eval_split])} примеров")
            
            return dataset, eval_split
            
        except Exception as e:
            print(f" Ошибка: {e}")
            # Создаем тестовый датасет
            train_data = {
                "text": ["Отличный продукт!", "Ужасное качество", "Нормально"] * 5000,
                "sentiment": [1, 2, 0] * 5000  
            }
            valid_data = {
                "text": ["Прекрасно", "Плохо", "Средне"] * 500,
                "sentiment": [1, 2, 0] * 500  
            }
            
            dataset = DatasetDict({
                'train': Dataset.from_dict(train_data),
                'validation': Dataset.from_dict(valid_data)
            })
            print(f"✅ Создан тестовый датасет: {len(dataset['train'])} train, {len(dataset['validation'])} validation")
            return dataset, 'validation'

    dataset, eval_split = load_dataset_fast()

    print("\n=== ПОДГОТОВКА ДАННЫХ ===")

    # ФУНКЦИЯ ПРЕОБРАЗОВАНИЯ МЕТОК
    def convert_labels_fast(example):
        sentiment = example.get('sentiment')
        if isinstance(sentiment, (int, float)):
            sentiment_int = int(sentiment)
            # 🎯 ПРЕОБРАЗОВАНИЕ для датасета (0=NEUTRAL, 1=POSITIVE, 2=NEGATIVE)
            if sentiment_int == 0:
                example['labels'] = 1  # NEUTRAL → 1
            elif sentiment_int == 1:
                example['labels'] = 2  # POSITIVE → 2
            elif sentiment_int == 2:
                example['labels'] = 0  # NEGATIVE → 0
            else:
                example['labels'] = 1  # NEUTRAL по умолчанию
        elif isinstance(sentiment, str):
            sentiment_lower = sentiment.lower()
            if any(word in sentiment_lower for word in ['negative', 'негатив', 'neg', 'плох', 'ужас']):
                example['labels'] = 0  # NEGATIVE
            elif any(word in sentiment_lower for word in ['positive', 'позитив', 'pos', 'хорош', 'отличн', 'прекрасн']):
                example['labels'] = 2  # POSITIVE
            else:
                example['labels'] = 1  # NEUTRAL
        else:
            example['labels'] = 1  # NEUTRAL по умолчанию
        return example

    print("🔄 преобразование меток...")
    dataset = dataset.map(convert_labels_fast)

    print(f"\n=== ЗАГРУЗКА МОДЕЛИ ===")

    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # 🔧 ЗАГРУЗКА МОДЕЛИ
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        id2label={0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"},
        label2id={"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    )
    print(f"✅ модель загружена: {config.model_name}")

    # 🔧ТОКЕНИЗАЦИЯ
    def tokenize_fast(examples):
        texts = [str(text) for text in examples["text"]]
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors=None
        )

    print("⚡ токенизация...")
    tokenized_datasets = dataset.map(
        tokenize_fast, 
        batched=True,
        batch_size=2000,  # 🚀 Очень большие батчи для скорости
        remove_columns=['text', 'sentiment']
    )

    print(f"📊 Данные готовы: {len(tokenized_datasets['train'])} train, {len(tokenized_datasets[eval_split])} eval")

    # 🔧 DATA COLLATOR
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 🔧 МЕТРИКИ
    def compute_metrics_fast(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro")
        }

    # 🔧 ОПТИМИЗИРОВАННЫЕ АРГУМЕНТЫ ОБУЧЕНИЯ (БЕЗ MULTIPROCESSING)
    training_args = TrainingArguments(
        output_dir="./fast-model-sentiment",
        overwrite_output_dir=True,
        
        # ⚡ МАКСИМАЛЬНАЯ СКОРОСТЬ
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation,
        fp16=config.fp16,
        
        # 🎯 БЫСТРОЕ ОБУЧЕНИЕ
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        
        # 📊 ВАЛИДАЦИЯ
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        
        # ⚡ ОПТИМИЗАЦИИ (ВЫКЛЮЧАЕМ MULTIPROCESSING ДЛЯ WINDOWS)
        dataloader_pin_memory=False,  # 🔧 ВЫКЛЮЧАЕМ для Windows
        dataloader_num_workers=0,     # 🔧 ВЫКЛЮЧАЕМ workers для Windows
        optim="adamw_torch",
        remove_unused_columns=True,
        label_names=["labels"],
        
        # 💾 СОХРАНЕНИЕ
        save_total_limit=2,
    )

    # 🔧 TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets[eval_split],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fast,
    )

    print(f"\n🎯 КОНФИГУРАЦИЯ МОДЕЛИ:")
    print(f"   Модель: {config.model_name}")
    print(f"   Train examples: {len(tokenized_datasets['train'])}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Effective batch: {config.batch_size * config.gradient_accumulation}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Max length: {config.max_length}")

    # 🔧 РАСЧЕТ ВРЕМЕНИ
    total_steps = len(tokenized_datasets['train']) * config.num_epochs / (config.batch_size * config.gradient_accumulation)
    estimated_time = total_steps / 50  # Примерно 200 шагов в минуту для  модели
    print(f"   Estimated time: ~{estimated_time:.1f} минут")

    print("\n🚀 ЗАПУСК БЫСТРОГО ОБУЧЕНИЯ...")

    try:
        import time
        start_time = time.time()
        
        # Обучаем с прогресс-баром
        train_result = trainer.train()
        
        end_time = time.time()
        training_time = (end_time - start_time) / 60
        
        print(f"✅ Обучение завершено за {training_time:.1f} минут!")
        print(f"📈 Final train loss: {train_result.metrics['train_loss']:.4f}")
        
        # 🔧 ФИНАЛЬНАЯ ОЦЕНКА
        print(f"\n📊 ФИНАЛЬНАЯ ОЦЕНКА НА {eval_split.upper()}:")
        eval_results = trainer.evaluate()
        for key, value in eval_results.items():
            print(f"   {key}: {value:.4f}")
        
        # Сохраняем модель
        trainer.save_model("./trained-distil-sentiment-model")
        print("💾 модель сохранена")
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()

    # 🔧 ТЕСТИРОВАНИЕ
    def predict_sentiment_fast(texts, model, tokenizer):
        model.eval()
        device = next(model.parameters()).device
        
        texts = [str(text) for text in texts]
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=config.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits.cpu(), dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        confidences = probabilities[torch.arange(len(texts)), predicted_classes]
        
        labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        
        return [
            {"text": text, "sentiment": labels[pred.item()], "confidence": conf.item()}
            for text, pred, conf in zip(texts, predicted_classes, confidences)
        ]

    print("\n=== ТЕСТ МОДЕЛИ ===")
    test_texts = [
        "Это прекрасный продукт! Очень доволен покупкой.",
        "Ужасное качество, никогда больше не куплю.",
        "Нормально, ничего особенного. Можно пользоваться.",
        "Восхитительно! Лучшее что я видел!",
        "Полный разочарование, зря потратил деньги."
    ]

    print("Тестируем на 5 примерах...")
    results = predict_sentiment_fast(test_texts, model, tokenizer)

    print("\n📊 РЕЗУЛЬТАТЫ ТЕСТА:")
    for i, result in enumerate(results, 1):
        emoji = "😊" if result["sentiment"] == "POSITIVE" else "😐" if result["sentiment"] == "NEUTRAL" else "😞"
        print(f"{i}. {emoji} {result['sentiment']:8} ({result['confidence']:.3f}) | {result['text'][:50]}...")


if __name__ == '__main__':
    main()