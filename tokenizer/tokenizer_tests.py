import regex as re
import tiktoken

from BasicTokenizer.py import BasicTokenizer
from RegexTokenizer.py import RegexTokenizer
from GPT4Tokenizer.py import GPT4Tokenizer

training_text = open('input.txt', 'r').read()
test_text = open('taylorswift.txt', 'r').read()

tokenizer = BasicTokenizer()
tokenizer.train(training_text, 20, verbose = True)

# simple tests
sample = tokenizer.encode(test_text)
print(sample[:100])
print(tokenizer.decode(sample[:1000]))

# Regex tokenizer
regex_tokenizer = RegexTokenizer()
regex_tokenizer.train(training_text, 20, verbose = True)

# simple tests
regex_sample = regex_tokenizer.encode(test_text)
print(sample[:100])
print(regex_tokenizer.decode(sample[:1000]))

# Check that regex and basic produce same result
print(regex_sample == sample)


enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
ids = enc.encode("hello world!!!? (안녕하세요!) lol123 😉")
text = enc.decode(ids) # get the same text back

gpt4tokenizer = GPT4Tokenizer()


enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
ids = enc.encode("hello world!!!? (안녕하세요!) lol123 😉")
text = enc.decode(ids) # get the same text back

print(ids)

our_ids = gpt4tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉")
our_text = gpt4tokenizer.decode(our_ids)

print(our_ids)

print(ids == our_ids)
print(text == our_text)
