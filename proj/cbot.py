from sentence_transformers.util import cos_sim
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import transformers
import os
import numpy as np
import pickle
import torch
import pandas as pd
import numpy as np
import random,string,re
import nltk
from nltk.corpus import stopwords
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
print("ctx_model")
ctx_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
print("qus_model")
question_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
stop_words = set(stopwords.words('english'))



x = pickle.load(open("C:/Users/HP/Documents/chatbot-project/file_ctx.pkl", 'rb')) #To load saved model from local directory

pooled_output = x 
xb = transformers.models.dpr.modeling_dpr.DPRContextEncoderOutput(torch.Tensor(x))
type(xb)
 

  
  
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def question_answer(question, text):
    
    #tokenize question and text in ids as a pair
    input_ids = tokenizer.encode(question, text)
    #print("input_ids : ",input_ids)
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    #print("tokens",tokens)
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    #print("sep_idx: ",sep_idx)
    #number of tokens in segment A - question
    num_seg_a = sep_idx+1
    #print("num_seg_a: ",num_seg_a)

    #number of tokens in segment B - text
    num_seg_b = len(input_ids) - num_seg_a
    #print("num_seg_b: ",num_seg_b)
    #list of 0s and 1s
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    #print("segment_ids: ",segment_ids)
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    #print("output iof the model",output)
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    #print(answer_start,"S:E",answer_end)
    answer = str()
    if answer_end >= answer_start:
        #print("inside the IF-print to check index of ans E>S ")
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    #else:           
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    
#     print("Text:\n{}".format(text.capitalize()))
#     print("\nQuestion:\n{}".format(question.capitalize()))
    #print("\nAnswer:\n{}".format(answer.capitalize()))
    return ("{}".format(answer.capitalize()))
    





def get_context(xq):
  res_lst =dict()
  for i, xq_vec in enumerate(xq.pooler_output):
        for root, dirs, files in os.walk(f"C:/Users/HP/Documents/BOT_SMCBOT/encoded_content/"):
            for file in files:
                print(file)
                filename, extension = os.path.splitext(file)
                if extension == '.pkl':
                    x = pickle.load(open(f"C:/Users/HP/Documents/BOT_SMCBOT/encoded_content/{filename}.pkl", 'rb'))
                    xb = transformers.models.dpr.modeling_dpr.DPRContextEncoderOutput(torch.Tensor(x))
                    print(xb.pooler_output)
                    probs = cos_sim(xq_vec, xb.pooler_output)
                    print("filename & result:",filename,"\n",probs)
                    print("filename & value",filename,"\n",torch.max(probs))
                    print("max",torch.max(probs))
                    print("\n\n\n")
                    res_lst[filename] = [torch.argmax(probs).item(),torch.max(probs)]
                
        filenm = next(iter(res_lst))
        v = list(res_lst.values())

        for key in res_lst:
            if res_lst[key][1] > res_lst[filenm][1]:
                filenm = key
                armx  = res_lst[key][0]
                
        print("FILE NAME",filenm,"\n ARGMAX",armx)
        with open(f"C:/Users/HP/Documents/BOT_SMCBOT/web content/{filenm}.txt",errors="ignore") as fs:
                const = fs.read().split("\n")
                if res_lst[filenm][1] < 0.6:
                    return None,None
                elif res_lst[filenm][1] >= 0.6 and res_lst[filenm][1] < 0.9:
                    return (const[armx], "same")
                else:
                    return (const[armx],"different")
                
                    #argmax = torch.argmax(probs)
      #print(context[argmax])
      #return context[argmax]
  
  

def basic_ques(q):   
    q = q.lower()
    l1 = ["Hi there","hi","Hola","Hello","Hello there","Hya","Hya there"]
    l2 = ["Hi there","Hola","Hi human, please tell me your qurries","Hello human, please tell me your qurries","Hola human, please tell me your qurries"]
    l3 = ["how are you","Hi how are you","hru","h r u","Hello how are you","Hola how are you","How are you doing","Hope you are doing well","Hello hope you are doing"]
    l4 = ["Hello, I am great, how are you? Please tell me your querries","Hello, how are you? I am great thanks! Please tell me your querries","Hello, I am good thank you, how are you? Please tell me your querries","Hi, I am great, how are you? Please tell me your querries","Hi, how are you? I am great thanks! Please tell me your querries","Hi, I am good thank you, how are you? Please tell me your querries","Hi, good thank you, how are you? Please tell me your querries"]
    l5 = ["What is your name","What could I call you","What can I call you","What do your friends call you","Who are you","Tell me your name","What is your real name","What is your real name please","What's your real name","Tell me your real name","Your real name","Your real name please","Your real name please"]
    l6 = ["You can call me SMCBOT","You may call me SMCBOT","Call me SMCBOT"]
    l7 = ["OK thank you","OK thanks","OK","Thanks","Thankyou","Thank you","That's helpful"]
    l8 = ["No problem!","Happy to help!","Any time!","My pleasure"]
    l9 = ["Thanks bye","bye","byeeeee","byyye","Thanks for the help goodbye","Thank you bye","Thank you, goodbye","Thanks goodbye","Thanks good bye"]
    l10 = ["No problem, goodbye","Not a problem! Have a nice day","Bye! Come back again soon."]
    resp = [[l1,l2],[l3,l4],[l5,l6],[l7,l8],[l9,l10]]
    for ar in resp:
        if q in [item.lower() for item in ar[0]]:
            ans = random.choice(ar[1])
            return ans
    return None


def puct_remove(strs):
    out = re.sub('[%s]' % re.escape(string.punctuation), '', strs)
    strs = ""
    for r in out.split():
        if  r.lower() not in stop_words:
            print("r:",r)
            strs += r 
    print(strs)
    return strs

     
def qna(question):
    question = puct_remove(question)
    b_ans = basic_ques(question)
    if b_ans == None:
        df = pd.read_csv("C:/Users/HP/Documents/BOT_SMCBOT/fdbkY.csv")
        for i in range(len(df)):
            if puct_remove(df.at[i,'Question'].lower()) == puct_remove(question.lower()):
                prev_ans = df.at[i,'Response']
                return prev_ans
            else:
                xq_tokens = question_tokenizer(question, max_length=256, padding='max_length', 
                                       truncation=True, return_tensors='pt')
                xq = question_model(**xq_tokens)
                text,chk = get_context(xq) 
                if chk == "different":
                    ans  = question_answer(question, text)
                elif chk == "same":
                    ans = text
                else:
                    ans = "unable to find answer"

                return ans
    else:
        return b_ans
    #print(qna("Where is the Postgraduate library located"))
    
  
from flask import Flask, render_template, request,json,jsonify
import csv

app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get",methods=['POST','GET'])
def get_bot_response():
    if request.method =='POST':
        data =request.values
        for d in data:
            d = json.loads(d)
            vals = list(d.values())
            if vals[-1] == 'yes':
                with open('fdbkY.csv','a+') as fb:
                    csv_wtr = csv.writer(fb)
                    csv_wtr.writerow(d.values())
                fb.close()
                return jsonify("success")
            elif vals[-1] =='no':
                with open('fdbkN.csv','a+') as fb:
                    csv_wtr = csv.writer(fb)
                    csv_wtr.writerow(d.values())
                fb.close()
                #return jsonify("success")
                #return redirect(url_for("main.home"))
                #return redirect('/get')
                return jsonify("success")

    else:
        userText = request.args.get('msg')
        return qna(userText)
    
    

    
      
if __name__ == "__main__":
    app.run(debug=True)
