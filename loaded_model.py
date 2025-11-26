import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline
)
import pandas as pd
from typing import List, Dict
import numpy as np

print("=== ТЕСТЕР МОДЕЛЕЙ ДЛЯ АНАЛИЗА ТОНАЛЬНОСТИ ===")

# КОНФИГУРАЦИЯ ТЕСТИРОВАНИЯ
class ModelTester:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.tokenizer = None
        self.model = None
        
    # 🎯 ДОСТУПНЫЕ МОДЕЛИ ДЛЯ ТЕСТИРОВАНИЯ
    def setup_models(self):
        """Настройка доступных моделей"""
        self.models = {
            "1": {
                "name": " Наша обученная RuBERT модель версия: standart",
                "path": "./trained-rubert-large-sentiment_standart",  # Путь к нашей обученной модели
                "type": "local"
            },
            "2": {
                "name": " Наша обученная RuBERT модель версия: slow",
                "path": "./trained-rubert-large-sentiment_slow",  # Путь к нашей обученной модели
                "type": "local"
            }
            # "3": {
            #      "name": "RuBERT Base (сбалансированная)", 
            #      "path": "sberbank-ai/ruBert-base",
            #      "type": "huggingface"
            #  },
            #  "4": {
            #      "name": "RuBERT Large (мощная)",
            #      "path": "sberbank-ai/ruBert-large", 
            #      "type": "huggingface"
            #  },
            #  "5": {
            #     "name": "RuRoberta Large",
            #      "path": "ai-forever/ruRoberta-large",
            #      "type": "huggingface"
            #  }
        }
    
    def print_available_models(self):
        """Показать доступные модели"""
        print("\n📚 ДОСТУПНЫЕ МОДЕЛИ:")
        for key, model_info in self.models.items():
            print(f"   {key}. {model_info['name']}")
            print(f"      📁 Путь: {model_info['path']}")
    
    def load_model(self, model_key: str):
        """Загрузка выбранной модели"""
        if model_key not in self.models:
            print(f"❌ Модель {model_key} не найдена!")
            return False
        
        model_info = self.models[model_key]
        print(f"\n🔄 Загружаем {model_info['name']}...")
        
        try:
            # Загружаем токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(model_info["path"])
            
            if model_info["type"] == "local":
                # Для локальных моделей указываем что это для классификации
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_info["path"],
                    num_labels=3  # 3 класса: negative, neutral, positive
                )
            else:
                # Для моделей с Hugging Face
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_info["path"],
                    num_labels=3
                )
            
            # Определяем устройство
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.current_model = model_key
            print(f"✅ {model_info['name']} успешно загружена на {self.device}!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def predict_sentiment(self, texts: List[str]) -> List[Dict]:
        """Предсказание тональности для списка текстов"""
        if self.model is None:
            print("❌ Модель не загружена!")
            return []
        
        try:
            # Токенизация
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            )
            
            # Перенос на устройство модели
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Обработка результатов
            probabilities = torch.nn.functional.softmax(outputs.logits.cpu(), dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidences = probabilities[torch.arange(len(texts)), predicted_classes]
            
            # Сопоставление с метками
            labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
            
            results = []
            for i, text in enumerate(texts):
                sentiment = labels[predicted_classes[i].item()]
                confidence = confidences[i].item()
                
                # Определяем эмодзи и цвет уверенности
                emoji = "😊" if sentiment == "POSITIVE" else "😐" if sentiment == "NEUTRAL" else "😞"
                
                if confidence > 0.8:
                    confidence_color = "🟢"  # Высокая уверенность
                elif confidence > 0.6:
                    confidence_color = "🟡"  # Средняя уверенность  
                else:
                    confidence_color = "🔴"  # Низкая уверенность
                
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "emoji": emoji,
                    "confidence_color": confidence_color
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return []
    
    def test_complex_sentences(self):
        """Тестирование на сложных предложениях"""
        
        # 🎭 ТЕСТОВЫЕ ПРЕДЛОЖЕНИЯ РАЗНОЙ СЛОЖНОСТИ
        test_sentences = [
            # 🟢 ПРОСТЫЕ И ЯСНЫЕ
            {"text": "Это просто великолепно! Очень доволен покупкой!", "expected": "POSITIVE"},
            {"text": "Ужасное качество, никогда больше не куплю!", "expected": "NEGATIVE"},
            {"text": "Нормальный товар, ничего особенного.", "expected": "NEUTRAL"},
            
            # 🟡 СРЕДНЕЙ СЛОЖНОСТИ  
            {"text": "В целом неплохо, но есть небольшие недочеты.", "expected": "NEUTRAL"},
            {"text": "Отличный продукт, жаль только что дороговато.", "expected": "POSITIVE"},
            {"text": "Не сказать что плохо, но и хорошего мало.", "expected": "NEGATIVE"},
            
            # 🔴 СЛОЖНЫЕ И ПРОТИВОРЕЧИВЫЕ
            {"text": "Ну просто замечательно... если не смотреть на цену.", "expected": "NEUTRAL"},
            {"text": "Качество отличное, но обслуживание подвело.", "expected": "NEUTRAL"},
            {"text": "Прекрасный сервис! Жаль, что товар не оправдал ожиданий.", "expected": "NEGATIVE"},
            
            # 🎭 САРКАЗМ И ИРОНИЯ
            {"text": "О да, просто восхитительно... ждать месяц за доставку.", "expected": "NEGATIVE"},
            {"text": "Ну конечно, отличное качество - сломалось через день.", "expected": "NEGATIVE"},
            {"text": "Просто супер, если вас устраивает низкое качество.", "expected": "NEGATIVE"},
            
            # 📚 ДЛИННЫЕ И СЛОЖНЫЕ
            {"text": "Несмотря на некоторые недостатки в сборке, общее впечатление от продукта остается положительным, поскольку основные функции работают стабильно и соответствуют заявленным характеристикам.", "expected": "POSITIVE"},
            {"text": "Хотя дизайн продукта действительно привлекательный и эргономичный, постоянные проблемы с программным обеспечением и низкая надежность компонентов существенно снижают общую оценку и заставляют сомневаться в целесообразности покупки.", "expected": "NEGATIVE"},
            
            # ❓ НЕОПРЕДЕЛЕННЫЕ
            {"text": "Не знаю даже что и сказать...", "expected": "NEUTRAL"},
            {"text": "Вроде бы неплохо, но как-то не очень.", "expected": "NEUTRAL"},
            {"text": "С одной стороны хорошо, с другой стороны плохо.", "expected": "NEUTRAL"}
        ]
        
        print("\n" + "="*80)
        print("🎭 ТЕСТИРОВАНИЕ НА СЛОЖНЫХ ПРЕДЛОЖЕНИЯХ")
        print("="*80)
        
        texts = [item["text"] for item in test_sentences]
        expected = [item["expected"] for item in test_sentences]
        
        results = self.predict_sentiment(texts)
        
        if not results:
            return
        
        # 📊 СТАТИСТИКА ТЕСТИРОВАНИЯ
        correct_predictions = 0
        confidence_sum = 0
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ({self.models[self.current_model]['name']}):")
        print("="*80)
        
        for i, (result, expected_sentiment) in enumerate(zip(results, expected)):
            is_correct = result["sentiment"] == expected_sentiment
            match_icon = "✅" if is_correct else "❌"
            
            if is_correct:
                correct_predictions += 1
            confidence_sum += result["confidence"]
            
            # Вывод результата
            print(f"{i+1:2d}. {match_icon} {result['emoji']} {result['confidence_color']}")
            print(f"    Предсказано: {result['sentiment']:8} (уверенность: {result['confidence']:.3f})")
            print(f"    Ожидалось:   {expected_sentiment:8}")
            print(f"    Текст: {result['text']}")
            print()
        
        # 📈 ИТОГОВАЯ СТАТИСТИКА
        accuracy = (correct_predictions / len(results)) * 100
        avg_confidence = (confidence_sum / len(results)) * 100
        
        print("📈 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   🎯 Точность: {accuracy:.1f}% ({correct_predictions}/{len(results)})")
        print(f"   💪 Средняя уверенность: {avg_confidence:.1f}%")
        print(f"   🔢 Протестировано предложений: {len(results)}")
        
        return accuracy, avg_confidence
    
    def interactive_test(self):
        """Интерактивное тестирование"""
        print("\n" + "="*60)
        print("💬 ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ")
        print("="*60)
        
        while True:
            print("\nВведите текст для анализа (или 'quit' для выхода):")
            user_text = input("📝 Ваш текст: ").strip()
            
            if user_text.lower() in ['quit', 'exit', 'выход']:
                break
            
            if not user_text:
                continue
            
            results = self.predict_sentiment([user_text])
            
            if results:
                result = results[0]
                print(f"\n🎯 РЕЗУЛЬТАТ:")
                print(f"   {result['emoji']} {result['sentiment']}")
                print(f"   {result['confidence_color']} Уверенность: {result['confidence']:.3f}")
                print(f"   💬 Текст: {user_text}")
            else:
                print("❌ Ошибка анализа текста")

# 🚀 ЗАПУСК ТЕСТЕРА
def main():
    tester = ModelTester()
    tester.setup_models()
    
    print("🎛️  УПРАВЛЕНИЕ ТЕСТЕРОМ МОДЕЛЕЙ")
    print("="*50)
    
    while True:
        print("\nВыберите действие:")
        print("1. 📊 Протестировать все модели на сложных предложениях")
        print("2. 🔄 Выбрать модель для тестирования")  
        print("3. 💬 Интерактивное тестирование")
        print("4. 🏆 Сравнить все модели")
        print("5. 🚪 Выход")
        
        choice = input("\nВаш выбор: ").strip()
        
        if choice == "1":
            # Тестируем все модели на сложных предложениях
            results = {}
            for model_key in tester.models.keys():
                print(f"\n{'='*60}")
                print(f"🧪 ТЕСТИРУЕМ: {tester.models[model_key]['name']}")
                print(f"{'='*60}")
                
                if tester.load_model(model_key):
                    accuracy, avg_confidence = tester.test_complex_sentences()
                    results[model_key] = {
                        "name": tester.models[model_key]["name"],
                        "accuracy": accuracy,
                        "avg_confidence": avg_confidence
                    }
            
            # Выводим сравнение всех моделей
            print("\n🏆 СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ:")
            print("="*80)
            for model_key, result in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
                print(f"📊 {result['name']}")
                print(f"   🎯 Точность: {result['accuracy']:.1f}%")
                print(f"   💪 Уверенность: {result['avg_confidence']:.1f}%")
                print()
        
        elif choice == "2":
            # Выбор конкретной модели
            tester.print_available_models()
            model_choice = input("\nВыберите модель (1-5): ").strip()
            if model_choice in tester.models:
                if tester.load_model(model_choice):
                    tester.test_complex_sentences()
                    tester.interactive_test()
            else:
                print("❌ Неверный выбор модели!")
        
        elif choice == "3":
            # Интерактивное тестирование
            if tester.current_model is None:
                print("❌ Сначала выберите модель!")
                continue
            tester.interactive_test()
        
        elif choice == "4":
            # Быстрое сравнение моделей
            print("\n⚡ БЫСТРОЕ СРАВНЕНИЕ МОДЕЛЕЙ:")
            quick_results = {}
            for model_key in tester.models.keys():
                if tester.load_model(model_key):
                    results = tester.predict_sentiment(["Отличный продукт!", "Ужасное качество", "Нормально"])
                    if results:
                        avg_conf = sum(r["confidence"] for r in results) / len(results)
                        quick_results[model_key] = {
                            "name": tester.models[model_key]["name"],
                            "avg_confidence": avg_conf
                        }
            
            for model_key, result in sorted(quick_results.items(), key=lambda x: x[1]["avg_confidence"], reverse=True):
                print(f"   {result['name']}: {result['avg_confidence']:.3f} ср. уверенность")
        
        elif choice == "5":
            print("👋 До свидания!")
            break
        
        else:
            print("❌ Неверный выбор!")

if __name__ == "__main__":
    main()