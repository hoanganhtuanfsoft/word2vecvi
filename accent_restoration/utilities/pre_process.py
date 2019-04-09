#!/usr/bin/env python
# coding: utf-8

import re

# extract block of words
# block of words is continuous words that separated by space character like comma, tab, dot or newline...
def extract_phrases(text):
  """ extract phrases, i.e. group of continuous words, from text """
  return re.findall(r' {0,1}\w[\w ]+', text, re.UNICODE)
