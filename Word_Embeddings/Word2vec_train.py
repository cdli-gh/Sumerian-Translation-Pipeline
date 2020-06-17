# Code to train a word2vec model with gensim
# For use with ml5.js word2vec examples
from gensim.models import Word2Vec
import re
import json
import sys
import argparse
import glob
import os

#Parsing for the user arguments
parser = argparse.ArgumentParser(description="Text File to Word2Vec Vectors")

#Required input file
parser.add_argument("input", help="Path to the input text file")

#Optional arguments (room for further extending the script's capabilities)
parser.add_argument("-o", "--output", default="vector.txt", help="Path to the output text file (default: vector.txt)")

args = parser.parse_args()

#Using the arguments from the arg dictionary
output_text_file = args.output

Training_txt_file=args.input

    
Monolingual_Text=[]
with open(Training_txt_file) as f:
    for line in f:
        line=line.strip()
        line=line.split()
        Monolingual_Text.append(line)
               
print()
print("Training Word2vec Model")
# Create the Word2Vec model
model = Word2Vec(Monolingual_Text, size=300, window=5, min_count=1, workers=4)
# Save the vectors to a text file
print()
print("Word2vec Model Trained")
model.wv.save_word2vec_format(output_text_file, binary=False)

'''# Open up that text file and convert to JSON
f = open(output_text_file)
v = {"vectors": {}}
for line in f:
    w, n = line.split(" ", 1)
    v["vectors"][w] = list(map(float, n.split()))

# Save to a JSON file
# Could make this an optional argument to specify output file
with open(output_text_file[:-4] + "json", "w") as out:
    json.dump(v, out)'''