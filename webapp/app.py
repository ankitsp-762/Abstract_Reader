import numpy as np
import gradio as gr
import tensorflow as tf
from model import get_model
from spacy.lang.en import English

model = get_model()

class_names = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']

#  not working with gradio
# def bold_text(text):
#     bold_start = '\033[1m'
#     bold_end = '\033[0m'
#     return bold_start + text + bold_end

def read(abstract_lines,test_abstract_pred_classes):
   seq = {
      'OBJECTIVE' : "",
      'BACKGROUND' : "",
      'METHODS' : "",
      'RESULTS' : "",
      'CONCLUSIONS' : ""
   }

   for i, line in enumerate(abstract_lines):
    seq[test_abstract_pred_classes[i]] += line

   result = ""
   for key in seq:
      result += key + ": "
      result += seq[key]
      result +='\n\n'

   return result
   

def split_chars(text):
  return " ".join(list(text))

def preprocess_predict(input):
    nlp = English()
    sentencizer = nlp.add_pipe("sentencizer")
    doc = nlp(str(input))
    abstract_lines = [str(sent) for sent in list(doc.sents)] 

    # Get total number of lines
    total_lines_in_sample = len(abstract_lines)

    # Go through each line in abstract and create a list of dictionaries containing features for each line
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample - 1
        sample_lines.append(sample_dict)

    test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)
    test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
    # One-hot encode to same depth as training data, so model accepts right input shape
    test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)
    # Split abstract lines into characters
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    # Make predictions on abstract features
    test_abstract_pred_probs = model.predict(x=(test_abstract_line_numbers_one_hot,
                                                   test_abstract_total_lines_one_hot,
                                                   tf.constant(abstract_lines),
                                                   tf.constant(abstract_chars)))
    
    test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
    # Turn prediction class integers into string class names
    test_abstract_pred_classes = [class_names[i] for i in test_abstract_preds]

    output = read(abstract_lines,test_abstract_pred_classes)
    return output
    


gr.Interface(fn=preprocess_predict, 
             inputs=gr.Textbox(lines=7, placeholder="Write Abstract..."),
             outputs="text",title = "Abstract Reader").launch()



