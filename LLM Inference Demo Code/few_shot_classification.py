# import packages
import os
import pandas as pd
from torch import bfloat16
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import joblib
from huggingface_hub import login
import torch
import re
from tqdm import tqdm
torch.cuda.empty_cache()

# function generate chat dictionary for titles
def data_preparation(df):
    # example 1 title
    title1 = "The Role of AI in Shaping Public Discourse"
    # example 2 title
    title2 = "Media Representations of Racial Identity: Challenges and Opportunities"
    # example 1 response
    res1 = "<code>0</code><reason>The title focuses on AI and its impact on public discourse, without explicit reference to race, racial identity, or related topics.</reason>"
    # example 2 response
    res2 = "<code>1</code><reason>The title explicitly addresses 'racial identity,' directly connecting to discussions of race and its representation in the media.</reason>"
    
    # info template
    info = ["Carefully analyze the following research title and determine whether it is related to race or not.",
            "Research is considered race-related if it explicitly mentions race or racial groups, racism, stereotypes, racial privilege, racial bias, or marginalization, or interactual perspectives in the context of race (e.g., diversity, culture, intercultural relations, or intercultural research), ethnic groups or ethnicity-based experiences in the context of citizenship (e.g., immigrants, migrants), and people (e.g., Egyptian, African, Chinese).",
            "Race-related does not include not geographic locations (e.g., Africa) unless tied to people or identity, religion (e.g., Muslims, Hindu) unless race/ethnicity is also explicitly mentioned, or individuals (e.g., Obama) unless their race or racial identity is explicitly mentioned.",
            "Return 0 in code tags if the title is not related to race, and 1 if it is related to race.",
            "Additionally, provide a brief explanation in the reason tags for the classification."
            ]
    info = ' '.join(info)
    info += '\n\nTitle:\n'

    info1 = info+title1
    info2 = info+title2
    info3 = info
    
    # chat template
    chat = [
        {"role": "system", "content": "You are a helpful assistant. Your task is to classify whether a title is related to race or not."},
        {"role": "user", "content": info1},
        {"role": "assistant", "content": res1},
        {"role": "user", "content": info2},
        {"role": "assistant", "content": res2},
        ]

    # list of chat templates for all titles
    chat_list = []
    for idx, row in df.iterrows():
        title = row['Title']
        info_dict = {}
        info_dict["role"] = "user"
        info = info3+str(title)
        info_dict["content"] = info
        new_chat = chat.copy()
        new_chat.append(info_dict)
        chat_list.append(new_chat)
    
    df['Chat'] = chat_list
    return df

# select any model from huggingface
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          cache_dir='/mnt/scratch/lellaom/models',
                                          padding_side="right",
                                          add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

# quantization configurations
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type='nf4', 
    bnb_4bit_use_double_quant=True,  
    bnb_4bit_compute_dtype=bfloat16  
)
# model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    cache_dir='/mnt/scratch/lellaom/models',  
)

# required for inference
model.eval()

# read input and prepare data
df = pd.read_csv("nca_pred_titles.csv")
df = df.iloc[0:10]
df = data_preparation(df)

# replace with your huggingface token
login(token="<hf_token>")

label = []
reason = []
# loop for label prediction
for chat in tqdm(df['Chat'], desc="Inference Running"):
    # convert chat to input_ids
    inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, 
                                           return_dict=True, return_tensors='pt').to(model.device)
    # predicted output_ids
    output_ids = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, do_sample=False, max_new_tokens=250)
    # generated text
    gen_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    gen_text = gen_text.split('[/INST]')[-1]
    # extract label from generated text
    match = re.search(r'\d', gen_text)
    if match:
        label.append(match.group(0))
    else:
        label.append(pd.NA)
    reason.append(gen_text)

# saving results
df = df.drop(columns=['Chat'])
df['Label'] = label
df['Reason'] = reason
df.to_csv(f"nca_pred_labels.csv", index=False)