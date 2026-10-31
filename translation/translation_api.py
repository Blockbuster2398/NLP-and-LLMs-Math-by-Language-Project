from os import environ
from google.cloud import translate

# See https://codelabs.developers.google.com/codelabs/cloud-translation-python3#0 for more information

# https://docs.cloud.google.com/translate/docs/translate-text#translate_v3_translate_text-python Check later for advanced details (model choice?)

# PROJECT_ID = environ.get("PROJECT_ID", "") # (Don't care about keeping this secret)
PROJECT_ID = "project-018451d2-a22e-40ab-821"
assert PROJECT_ID
PARENT = f"projects/{PROJECT_ID}"

def print_supported_languages(display_language_code: str):
    client = translate.TranslationServiceClient()
    response = client.get_supported_languages(
        parent=PARENT,
        display_language_code=display_language_code,
    )
    languages = response.languages
    print(f" Languages: {len(languages)} ".center(60, "-"))
    for language in languages:
        language_code = language.language_code
        display_name = language.display_name
        print(f"{language_code:10}{display_name}")

def translate_text(text: str, target_language_code: str) -> translate.Translation:
    client = translate.TranslationServiceClient()
    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        target_language_code=target_language_code,
        mime_type="text/plain",  # default is HTML, which escapes apostrophes etc. (&#39;)
    )
    return response.translations[0]

def detect_language(text: str) -> translate.DetectedLanguage:
    client = translate.TranslationServiceClient()
    response = client.detect_language(parent=PARENT, content=text)
    return response.languages[0]

    
if __name__ == "__main__":
    
    # Supported Languages
    print_supported_languages("en")

    # Translation Demo
    text = "Hello World!"
    target_languages = ["tr", "de", "es", "it", "el", "zh", "ja", "ko"]
    print(f" {text} ".center(50, "-"))
    for target_language in target_languages:
        translation = translate_text(text, target_language)
        source_language = translation.detected_language_code
        translated_text = translation.translated_text
        print(f"{source_language} → {target_language} : {translated_text}")

    # Language Detection Demo
    sentences = [
        "Selam Dünya!",
        "Hallo Welt!",
        "¡Hola Mundo!",
        "Ciao mondo!",
        "Γεια σου Κόσμο!",
        "你好世界！",
        "「こんにちは世界」",
        "안녕하세요!",
    ]
    for sentence in sentences:
        language = detect_language(sentence)
        confidence = language.confidence
        language_code = language.language_code
        print(
            f"Confidence: {confidence:4.0%}",
            f"Language: {language_code:5}",
            sentence,
            sep=" | ",
        )