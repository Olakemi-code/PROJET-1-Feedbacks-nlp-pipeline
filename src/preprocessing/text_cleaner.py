import re
import unicodedata
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Ensure UTF-8 encoding
    text = text.encode('utf-8', 'ignore').decode('utf-8')

    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text