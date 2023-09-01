#Imported libraries
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


#Model initiation
device = "cuda"
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

def get_perplexity_of_line(lines):
  # Tokenize the input lines using the provided tokenizer and convert them to PyTorch tensors
  encodings = tokenizer(lines, return_tensors="pt")
  # Get the maximum sequence length allowed by the model's configuration
  max_length = model.config.n_positions
  # Define the stride for iterating through the sequence
  stride = 512
  # Get the length of the tokenized input sequence
  seq_len = encodings.input_ids.size(1)
  # Initialize a list to store negative log likelihoods (nlls)
  nlls = []
  # Initialize the previous end location for tracking sequence segments
  prev_end_loc = 0
  # Iterate through the sequence with a sliding window approach
  for begin_loc in tqdm(range(0, seq_len, stride)):
      # Calculate the end location of the current segment
      end_loc = min(begin_loc + max_length, seq_len) 
      # Calculate the target length for the autoregressive loss
      trg_len = end_loc - prev_end_loc  
      # Extract the input tokens for the current segment and move them to the specified device
      input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)   
      # Clone the input_ids tensor to create target_ids for the autoregressive loss
      target_ids = input_ids.clone()    
      # Mask out the future tokens in the target_ids by setting them to -100
      target_ids[:, :-trg_len] = -100
      # Calculate the model's output and the corresponding negative log likelihood loss
      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)
          neg_log_likelihood = outputs.loss
      # Append the negative log likelihood loss to the nlls list
      nlls.append(neg_log_likelihood)
      # Update the previous end location for the next iteration
      prev_end_loc = end_loc     
      # Check if the end of the sequence has been reached
      if end_loc == seq_len:
          break
  # Calculate the perplexity by taking the exponential of the mean of the negative log likelihoods
  ppl = torch.exp(torch.stack(nlls).mean())
  return ppl

def split_lines(sentence):
  offset =""
  perplexity_per_line = []
  filtered_lines = []
  lines = re.split(r"(?<=[.?!][ \[\(])|(?<=\n)\s*", sentence) #Splits a given sentence into a list of lines by identifying the positions after punctuation marks followed by whitespace or newlines
  lines = list(filter(lambda x: (x is not None) and (len(x) > 0), lines)) #Uses built in filter function, and lamda function, check if element x is not none, and length greater than 0. List converts back to iterator
  lines = [re.sub(r"^[•\-–—]\s*", "", line) for line in lines]  # Remove bullet points (•, -, –, —)
  lines = [line.strip() for line in lines]  # Remove leading/trailing whitespace
  lines = [line.replace("\r", "") for line in lines] #remove return carriage
  for i, line in enumerate(lines):
    #This checks if a line does not contain any letters or digits and skips to the next iteration if it doesn't.
    if re.search("[a-zA-Z0-9]+", line) == None:
      continue
    #Check if offset is greater than zero, if it is concats it together with line, before setting offset to empty once again.
    if len(offset) > 0:
      line = offset + line
      offset =""
    #Checks the first character of the variable line to see if it is newline or space character. If it is, then it removes the first char from line by slicing.
    if line[0] == "\n" or line[0] == " ":
      line = line[1:]
    #This code checks if the last character of variable line is newline or space character, if it is then it removes it by slicing from beginning to second last char.
    if line[-1] == "\n" or line[-1] == " ":
      line = line[:-1]
    #This checks if the last character of line, is open square bracket or open round bracket. If it is, then it removes it via slicing from begining to second last char.
    elif line[-1] == "[" or line[-1] == "(":
      offset = line[-1]
      line = line[:-1]
    filtered_lines.append(line)
    perplexityGrade = get_perplexity_of_line(line)
    perplexityGrade = round(float(perplexityGrade), 3)
    perplexity_per_line.append(perplexityGrade)
  return perplexity_per_line, filtered_lines

def average_line_perplexity(perplexity_per_line):
  try:
    removed_nan_values = [x for x in perplexity_per_line if not np.isnan(x)]
    average = sum(removed_nan_values) / len(removed_nan_values)
    average = round(average, 3)
  except:
    average = 0
  print("Average Perplexity of Text:", average)
  return average

def thresholds(averagePerplexity):
  if averagePerplexity < 90:
    value = 0
  elif averagePerplexity < 110:
    value = 0.5
  else:
    value = 1
  return value

  