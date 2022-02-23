"""
Train a model on TACRED.
"""

from stanfordcorenlp import StanfordCoreNLP
from functools import reduce
import concurrent.futures as fu
import copy

nlp = StanfordCoreNLP("/home/lijijie/StanfordNLP/stanford-corenlp-latest/stanford-corenlp-4.1.0")

a = nlp.pos_tag("Laura like the phone, which is sent by her husband.")
b = 3