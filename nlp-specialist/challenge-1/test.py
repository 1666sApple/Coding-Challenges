import re
from googletrans import Translator

class CodeMixedLanguageSeparator:
    def __init__(self, transliterate=False):
        self.transliterate = transliterate
        self.translator = Translator()
        
        # Regex patterns for each category
        self.bangla_script_pattern = re.compile(r'[\u0980-\u09FF]+')  # Unicode range for Bangla script
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        self.romanized_bangla_pattern = re.compile(r'[a-z]+', re.IGNORECASE)  # Catch possible romanized Bangla
    
    def detect_and_separate(self, text):
        # Initialize containers for the separated text
        bangla_script = []
        english = []
        romanized_bangla = []
        
        # Split the text into words to classify them
        words = text.split()
        
        for word in words:
            if self.bangla_script_pattern.search(word):
                bangla_script.append(word)
            elif self.english_pattern.search(word):
                # Check if it's purely English or possibly Romanized Bangla
                if self.is_romanized_bangla(word):
                    romanized_bangla.append(word)
                else:
                    english.append(word)
        
        # Transliterate romanized Bangla if required
        romanized_transliterated = []
        if self.transliterate:
            for word in romanized_bangla:
                romanized_transliterated.append(self.transliterate_bangla(word))
        
        result = {
            "bangla_script": ' '.join(bangla_script),
            "english": ' '.join(english),
            "romanized_bangla": ' '.join(romanized_bangla),
        }
        
        if self.transliterate:
            result["romanized_transliterated"] = ' '.join(romanized_transliterated)
        
        return result
    
    def is_romanized_bangla(self, word):
        # Detect Romanized Bangla using basic phonetic patterns
        romanized_bangla_phonetics = re.compile(r'((bh|ch|kh|sh|th|dh|ph)|[aeiouy]{1,2}[bdgkptmnlrsv]{0,1})', re.IGNORECASE)

        # A word is likely Romanized Bangla if it matches typical Bangla phonetic patterns
        if romanized_bangla_phonetics.search(word):
            return True
        return False
    
    def transliterate_bangla(self, word):
        # Use Google Translate's API to transliterate from Romanized Bangla to Bangla script
        translation = self.translator.translate(word, src='en', dest='bn')
        return translation.text

# Sample usage
text = "Amar sonar Bangla, I love you. তোমার আকাশ তোমার বাতাস আমার prane bajay bashi."
separator = CodeMixedLanguageSeparator(transliterate=True)
result = separator.detect_and_separate(text)
print(result)
