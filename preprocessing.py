import re

def clean_text(text):
  #remove space
  text = text.strip()

  #lower all words into lower case
  text = text.lower()

  #remove hashtags
  text = text.replace("#", '')

  #remove urls "https//"
  text = re.sub(r"http\S+", "", text)

  #remove mentions @
  text = re.sub(r"@\w+", "", text)

  #remove numbers or punctuations
  text = re.sub(r"[^a-z\s]", "", text)

  return text