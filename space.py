import spacy

# pip install spacy
# python -m spacy download en_core_web_md
model = spacy.load("en_core_web_md")

sentence1 = model('Bears are cool.')
sentence2 = model('Your mom is a menace to society')

print(sentence1.similarity(sentence2))