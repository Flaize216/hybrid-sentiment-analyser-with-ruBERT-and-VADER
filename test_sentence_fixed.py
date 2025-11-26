import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import warnings

warnings.filterwarnings("ignore")

print("=== 🎭 ГИБРИДНЫЙ АНАЛИЗАТОР ТОНАЛЬНОСТИ ===")
print("🔧 Vader (английский) + RuBERT (русский)")


class HybridSentimentAnalyzer:
    def __init__(
        self,
        rubert_model_path: str = "./trained-rubert-large-sentiment_slow",
        vader_confidence_threshold: float = 0.65,
    ):
        """
        Инициализация гибридного анализатора

        Args:
            rubert_model_path: Путь к обученной модели RuBERT
            vader_confidence_threshold: Порог уверенности Vader (0-1)
        """
        print("🔄 Инициализация анализаторов...")

        # 🔧 СОХРАНЯЕМ ПОРОГ КАК АТРИБУТ КЛАССА
        self.vader_confidence_threshold = vader_confidence_threshold
        print(
            f"✅ Установлен порог уверенности Vader: {self.vader_confidence_threshold}"
        )

        # 🔧 ИНИЦИАЛИЗАЦИЯ VADER (для английского)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        print("✅ Vader Sentiment Analyzer загружен")

        # 🔧 ИНИЦИАЛИЗАЦИЯ ПЕРЕВОДЧИКА
        self.translator = GoogleTranslator(source="ru", target="en")
        print("✅ Переводчик GoogleTranslator инициализирован")

        # 🔧 ИНИЦИАЛИЗАЦИЯ RuBERT (для русского)
        try:
            self.rubert_tokenizer = AutoTokenizer.from_pretrained(rubert_model_path)
            self.rubert_model = AutoModelForSequenceClassification.from_pretrained(
                rubert_model_path, num_labels=3
            )

            # Определяем устройство и переносим модель
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.rubert_model = self.rubert_model.to(self.device)
            self.rubert_model.eval()

            print(f"✅ RuBERT модель загружена на {self.device}")
            print(f"📁 Путь к модели: {rubert_model_path}")

        except Exception as e:
            print(f"❌ Ошибка загрузки RuBERT модели: {e}")
            print("🔄 Пробуем загрузить стандартную модель...")
            self._load_fallback_rubert()

    def set_vader_threshold(self, new_threshold: float):
        """
        Установка нового порога уверенности Vader

        Args:
            new_threshold: Новый порог (0.0-1.0)
        """
        if 0 <= new_threshold <= 1:
            old_threshold = self.vader_confidence_threshold
            self.vader_confidence_threshold = new_threshold
            print(f"✅ Порог Vader изменен: {old_threshold} → {new_threshold}")
            return True
        else:
            print(f"❌ Порог должен быть между 0.0 и 1.0")
            return False

    def get_current_threshold(self) -> float:
        """Получить текущий порог уверенности"""
        return self.vader_confidence_threshold

    def _load_fallback_rubert(self):
        """Загрузка резервной модели RuBERT"""
        try:
            self.rubert_tokenizer = AutoTokenizer.from_pretrained(
                "sberbank-ai/ruBert-large"
            )
            self.rubert_model = AutoModelForSequenceClassification.from_pretrained(
                "sberbank-ai/ruBert-large", num_labels=3
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.rubert_model = self.rubert_model.to(self.device)
            self.rubert_model.eval()
            print("✅ Резервная RuBERT модель загружена")
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            raise

    def translate_text(self, text: str) -> str:
        """
        Перевод русского текста на английский

        Args:
            text: Русский текст для перевода

        Returns:
            Переведенный английский текст
        """
        try:
            translated = self.translator.translate(text)
            return translated
        except Exception as e:
            print(f"❌ Ошибка перевода: {e}")
            return text  # Возвращаем оригинальный текст в случае ошибки

    def vader_analyze(self, text: str) -> dict:
        """
        Анализ тональности с помощью Vader (для английского текста)

        Args:
            text: Английский текст для анализа

        Returns:
            Словарь с результатами анализа Vader
        """
        try:
            # Получаем оценки от Vader
            scores = self.vader_analyzer.polarity_scores(text)

            # Определяем основную тональность
            compound = scores["compound"]

            if compound >= 0.05:
                sentiment = "POSITIVE"
                confidence = compound  # Для позитивных используем compound
            elif compound <= -0.05:
                sentiment = "NEGATIVE"
                confidence = abs(
                    compound
                )  # Для негативных используем абсолютное значение
            else:
                sentiment = "NEUTRAL"
                # Для нейтральных используем разницу от 0
                confidence = 1 - min(abs(compound), 0.05) * 20

            return {
                "sentiment": sentiment,
                "confidence": min(confidence, 1.0),  # Ограничиваем до 1.0
                "scores": scores,
                "analyzer": "Vader",
            }

        except Exception as e:
            print(f"❌ Ошибка Vader анализа: {e}")
            return None

    def rubert_analyze(self, text: str) -> dict:
        """
        Анализ тональности с помощью RuBERT (для русского текста)

        Args:
            text: Русский текст для анализа

        Returns:
            Словарь с результатами анализа RuBERT
        """
        try:
            # Токенизация
            inputs = self.rubert_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=256
            )

            # Перенос на устройство модели
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Предсказание
            with torch.no_grad():
                outputs = self.rubert_model(**inputs)

            # Обработка результатов
            probabilities = torch.nn.functional.softmax(outputs.logits.cpu(), dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()

            # Сопоставление с метками
            labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
            sentiment = labels[predicted_class_id]

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "analyzer": "RuBERT",
            }

        except Exception as e:
            print(f"❌ Ошибка RuBERT анализа: {e}")
            return None

    def hybrid_analyze(self, russian_text: str, custom_threshold: float = None) -> dict:
        """
        Гибридный анализ тональности

        Args:
            russian_text: Исходный русский текст
            custom_threshold: Кастомный порог (если None, используется установленный)

        Returns:
            Словарь с результатами гибридного анализа
        """
        # 🔧 ИСПОЛЬЗУЕМ КАСТОМНЫЙ ПОРОГ ИЛИ УСТАНОВЛЕННЫЙ
        threshold = (
            custom_threshold
            if custom_threshold is not None
            else self.vader_confidence_threshold
        )

        print(f"\n🔍 АНАЛИЗ ТЕКСТА: '{russian_text}'")
        print("=" * 60)
        print(f"🎯 Текущий порог Vader: {threshold:.2f}")

        # 📝 ШАГ 1: Анализ RuBERT (оригинальный русский текст)
        print("🔄 RuBERT анализирует оригинальный текст...")
        rubert_result = self.rubert_analyze(russian_text)

        if rubert_result:
            print(
                f"   ✅ RuBERT: {rubert_result['sentiment']} (уверенность: {rubert_result['confidence']:.3f})"
            )
        else:
            print("   ❌ RuBERT: ошибка анализа")

        # 🌐 ШАГ 2: Перевод и анализ Vader
        print("🔄 Перевод на английский для Vader...")
        english_text = self.translate_text(russian_text)
        print(f"   📖 Перевод: '{english_text}'")

        print("🔄 Vader анализирует перевод...")
        vader_result = self.vader_analyze(english_text)

        if vader_result:
            print(
                f"   ✅ Vader: {vader_result['sentiment']} (уверенность: {vader_result['confidence']:.3f})"
            )
            print(f"   📊 Vader scores: {vader_result['scores']}")
        else:
            print("   ❌ Vader: ошибка анализа")

        # 🎯 ШАГ 3: Принятие решения
        print(f"\n🎯 ПРИНЯТИЕ РЕШЕНИЯ (порог: {threshold:.2f}):")

        final_result = {
            "original_text": russian_text,
            "translated_text": english_text,
            "final_sentiment": None,
            "final_confidence": None,
            "used_analyzer": None,
            "vader_result": vader_result,
            "rubert_result": rubert_result,
            "decision_reason": None,
            "threshold_used": threshold,
        }

        # Если Vader уверен больше порога - используем его результат
        if vader_result and vader_result["confidence"] >= threshold:
            final_result["final_sentiment"] = vader_result["sentiment"]
            final_result["final_confidence"] = vader_result["confidence"]
            final_result["used_analyzer"] = "Vader"
            final_result["decision_reason"] = (
                f"Vader уверен на {vader_result['confidence']:.1%} (порог: {threshold:.0%})"
            )
            print(f"   ✅ Используем Vader: {final_result['decision_reason']}")

        # Иначе используем RuBERT
        elif rubert_result:
            final_result["final_sentiment"] = rubert_result["sentiment"]
            final_result["final_confidence"] = rubert_result["confidence"]
            final_result["used_analyzer"] = "RuBERT"
            final_result["decision_reason"] = (
                f"Vader недостаточно уверен ({vader_result['confidence']:.1%} при пороге {threshold:.0%})"
            )
            print(f"   ✅ Используем RuBERT: {final_result['decision_reason']}")

        else:
            final_result["final_sentiment"] = "NEUTRAL"
            final_result["final_confidence"] = 0.5
            final_result["used_analyzer"] = "Fallback"
            final_result["decision_reason"] = (
                "Оба анализатора не сработали, используем нейтральный результат"
            )
            print("   ⚠️  Используем fallback: оба анализатора не сработали")

        return final_result

    def print_analysis_result(self, result: dict):
        """Красивый вывод результатов анализа"""
        print("\n" + "🎯 РЕЗУЛЬТАТ АНАЛИЗА " + "=" * 40)

        # Эмодзи для тональности
        emoji_map = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}

        emoji = emoji_map.get(result["final_sentiment"], "❓")
        confidence_color = (
            "🟢"
            if result["final_confidence"] > 0.7
            else "🟡" if result["final_confidence"] > 0.5 else "🔴"
        )

        print(f"📝 Оригинальный текст: {result['original_text']}")
        print(f"🌐 Перевод: {result['translated_text']}")
        print(f"\n{emoji} ФИНАЛЬНАЯ ТОНАЛЬНОСТЬ: {result['final_sentiment']}")
        print(f"{confidence_color} УВЕРЕННОСТЬ: {result['final_confidence']:.3f}")
        print(f"🔧 ИСПОЛЬЗОВАН: {result['used_analyzer']}")
        print(f"🎯 ПОРОГ: {result['threshold_used']:.2f}")
        print(f"💡 ПРИЧИНА: {result['decision_reason']}")

        # Детали анализаторов
        print(f"\n📊 ДЕТАЛИ АНАЛИЗАТОРОВ:")
        if result["vader_result"]:
            vader_emoji = emoji_map.get(result["vader_result"]["sentiment"], "❓")
            print(
                f"   Vader: {vader_emoji} {result['vader_result']['sentiment']} (уверенность: {result['vader_result']['confidence']:.3f})"
            )

        if result["rubert_result"]:
            rubert_emoji = emoji_map.get(result["rubert_result"]["sentiment"], "❓")
            print(
                f"   RuBERT: {rubert_emoji} {result['rubert_result']['sentiment']} (уверенность: {result['rubert_result']['confidence']:.3f})"
            )

        print("=" * 60)


# 🎯 ФУНКЦИЯ ДЛЯ ТЕСТИРОВАНИЯ С РАЗНЫМИ ПОРОГАМИ
def test_with_different_thresholds():
    """Тестирование с разными порогами уверенности"""

    analyzer = HybridSentimentAnalyzer()

    test_texts = [
        "Это просто великолепно! Очень доволен покупкой!",
        "Ужасное качество, никогда больше не куплю!",
        "Нормальный товар, ничего особенного.",
        "В целом неплохо, но есть небольшие недочеты.",
    ]

    thresholds = [0.5, 0.65, 0.8, 0.9]  # Разные пороги для тестирования

    for threshold in thresholds:
        print(f"\n🧪 ТЕСТ С ПОРОГОМ: {threshold}")
        print("=" * 50)

        analyzer.set_vader_threshold(threshold)

        for i, text in enumerate(test_texts, 1):
            print(f"\n📋 Пример {i}: '{text}'")
            result = analyzer.hybrid_analyze(text)

            # Краткий вывод для сравнения
            vader_conf = (
                result["vader_result"]["confidence"] if result["vader_result"] else 0
            )
            used_analyzer = result["used_analyzer"]
            print(
                f"   Vader уверенность: {vader_conf:.3f}, Использован: {used_analyzer}"
            )


# 💬 ИНТЕРАКТИВНЫЙ РЕЖИМ
def interactive_mode():
    """Интерактивный режим для ввода пользовательских текстов"""
    analyzer = HybridSentimentAnalyzer()

    print("\n💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 50)
    print(f"Текущий порог Vader: {analyzer.get_current_threshold()}")
    print("Вводите русские тексты для анализа тональности")
    print("Команды: 'threshold' - изменить порог, 'quit' - выход")
    print("=" * 50)

    while True:
        user_input = input("\n📝 Введите русский текст или команду: ").strip()

        if user_input.lower() in ["quit", "exit", "выход"]:
            print("👋 До свидания!")
            break

        elif user_input.lower() == "threshold":
            try:
                new_threshold = float(
                    input("Введите новый порог уверенности Vader (0.0-1.0): ")
                )
                if analyzer.set_vader_threshold(new_threshold):
                    print(f"✅ Новый порог установлен: {new_threshold}")
                else:
                    print("❌ Не удалось установить порог")
            except ValueError:
                print("❌ Введите число")
            continue

        elif not user_input:
            print("⚠️  Пожалуйста, введите текст")
            continue

        # Анализируем текст
        result = analyzer.hybrid_analyze(user_input)
        analyzer.print_analysis_result(result)


# 🚀 ОСНОВНАЯ ФУНКЦИЯ
def main():
    print("🎭 ГИБРИДНЫЙ АНАЛИЗАТОР ТОНАЛЬНОСТИ")
    print("🔧 Vader (английский перевод) + RuBERT (русский оригинал)")
    print("=" * 60)

    # Создаем анализатор с начальным порогом
    analyzer = HybridSentimentAnalyzer(vader_confidence_threshold=0.65)

    while True:
        print(f"\nТекущий порог Vader: {analyzer.get_current_threshold()}")
        print("\nВыберите режим:")
        print("1. 🧪 Тестирование на примерах")
        print("2. 💬 Интерактивный режим")
        print("3. ⚙️  Настроить порог уверенности Vader")
        print("4. 📊 Тест с разными порогами")
        print("5. 🚪 Выход")

        choice = input("\nВаш выбор: ").strip()

        if choice == "1":
            test_texts = [
                "Это просто великолепно! Очень доволен покупкой!",
                "Ужасное качество, никогда больше не куплю!",
                "Нормальный товар, ничего особенного.",
            ]

            for text in test_texts:
                result = analyzer.hybrid_analyze(text)
                analyzer.print_analysis_result(result)

        elif choice == "2":
            interactive_mode()

        elif choice == "3":
            try:
                new_threshold = float(
                    input("Введите новый порог уверенности Vader (0.0-1.0): ")
                )
                if analyzer.set_vader_threshold(new_threshold):
                    print(f"✅ Новый порог установлен: {new_threshold}")
                else:
                    print("❌ Не удалось установить порог")
            except ValueError:
                print("❌ Введите число")

        elif choice == "4":
            test_with_different_thresholds()

        elif choice == "5":
            print("👋 До свидания!")
            break

        else:
            print("❌ Неверный выбор!")


if __name__ == "__main__":
    main()
