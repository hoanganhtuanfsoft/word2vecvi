import config 
import emoji, re
import csv

from accent_restoration import accent_pipeline
from underthesea import word_tokenize
from langdetect import detect 

# Remove all sentences that contain exception character.
def remove_exception(context):
    result = ""
    sents = context.split("\n")
    for i, sent in enumerate(sents):
        if sent.find(chr(769)) != -1 or \
            sent.find(chr(768)) != -1 or \
                sent.find(chr(803)) != -1 or \
                    sent.find(chr(777)) != -1 or \
                        sent.find(chr(771)) != -1:
                        continue
        result += sent+"\n"

    return result
# Change teen code to right word
# Input from file with the following format:
# [teen_word,correct_word]
def change_teen_code(content):
    # Read teen code vocabulary 
    lst_teen_code = [] 
    with open(config.preprocessing.teen_code_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            lst_teen_code.append(" {} ".format(row[0]))
            lst_teen_code.append(" {} ".format(row[1].split("+")[0]))

    if len(lst_teen_code) % 2 != 0:
        # Remove the last word.
        lst_teen_code = lst_teen_code[:-1]

    # Add leading and trailing space with each line
    content = content.split("\n")
    content = "\n".join([" {} ".format(x) for x in content if x !=""])
    # Change teen code to original
    for i in range(0,len(lst_teen_code),2):
        content = content.replace(lst_teen_code[i],lst_teen_code[i+1])
    
    # Remove leading and trailing space
    content = content.split("\n")
    content = "\n".join([x.strip() for x in content if x !=""])
    return content

def normalize_content(content):
    #Remove các ký tự kéo dài: vd: đẹppppppp
    content = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), content, flags=re.IGNORECASE).lower()

    content = content.split("\n")
    # Keep all line that has [minimum_number_of_words] words or more.
    content = [x.strip() for x in content if x !="" and len(x.split()) >= config.preprocessing.minimum_number_of_words]

    return "\n".join(content)

def remove_emoji(content):
    return "".join([c for c in content if c not in emoji.UNICODE_EMOJI])

def spelling_correction(content):
    content = content.split("\n")
    content = [accent_pipeline.accent_restore(x) for x in content]

    # Apply word tokenize for each sentence
    result = []
    for sent in content:
        words = word_tokenize(sent)
        sentence = " ".join([x.replace(" ","_") for x in words])
        result.append(sentence)
    
    return "\n".join(result)

def remove_other_language(content):
    content = content.split("\n")
    result = []
    for sent in content:
        try:
            if detect(sent) =="vi":
                result.append(sent)
        except:
            pass 

    return "\n".join(result)



if __name__ == "__main__":
    input = "./data/data_ver03.txt"
    output = "./data/data_offical.txt"
    content = None 
    with open(input,"r") as f:
        content = f.read()

    content = normalize_content(content)

    content = change_teen_code(content)

    content = remove_emoji(content)

    content = remove_other_language(content)

    # word tokenize again.
    content = content.replace("_"," ").split("\n")
    
    result = []
    for sent in content:
        words = word_tokenize(sent)
        words = [x.replace(" ","_") for x in words if x !=""]
        result.append(" ".join(words))

    result = "\n".join(result)

    with open(output,"w") as f:
        f.write(result)



