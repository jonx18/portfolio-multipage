import streamlit as st
import pickle
from pathlib import Path
from fastai.text.all import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import wandb
from fastai.callback.wandb import *
from utils import get_unique_file, get_unique_directory
import ast
import SessionState
import pandas as pd

def tokenize(text):
    toks = tokenizer.tokenize(text)#, max_length=1024,truncation=True)
    return tensor(tokenizer.convert_tokens_to_ids(toks)).long()

class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]

class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        return x if isinstance(x, Tensor) else tokenize(x)
        
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))

def load_models():
    pretrained_weights = 'jonx18/DialoGPT-small-Creed-Odyssey'
    model = AutoModelForCausalLM.from_pretrained(pretrained_weights)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
    # classifier = pipeline("zero-shot-classification",
    #                   model=AutoModelForSequenceClassification.from_pretrained(get_unique_directory('trained_model_bart_large_mnli')),
    #                   tokenizer=AutoTokenizer.from_pretrained(get_unique_directory('trained_model_bart_large_mnli')))
    classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    with open(get_unique_file('tokenized.pkl'),"rb") as file:
        tokenized = pickle.load(file)
    bs,sl = 4,256
    tls = TfmdLists(tokenized, TransformersTokenizer(tokenizer), splits=RandomSplitter(seed=777,valid_pct=0.1)(tokenized), dl_type=LMDataLoader)
    dls = tls.dataloaders(bs=bs, seq_len=sl)
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()
    return learn, tokenizer, classifier

def get_preds(prompt, learn, tokenizer, do_sample=True, max_length=50,
              top_k=50, top_p=0.95, num_return_sequences=3,
              candidate_labels=[], multi_class=True ):
    candidate_labels.sort()
    prompt_ids = tokenizer.encode(prompt)
    inp = tensor(prompt_ids)[None]
    preds = learn.model.generate(inp,
        do_sample=do_sample, 
        max_length=max_length, 
        top_k=top_k, 
        top_p=top_p, 
        num_return_sequences=num_return_sequences)
    return tokenizer.decode(preds[0].cpu().numpy(), skip_special_tokens=True)


def get_classification(prompt, classifier, candidate_labels=['Positive','Negative'], multi_class=True ):
    lines = prompt.splitlines()
    results = []
    for line in lines:
        parts = line.split(':')
        if len(parts)<2: continue
        output_classifier = classifier(parts[1], candidate_labels, multi_class=multi_class)
        output_classifier['sequence'] = line 
        results.append(output_classifier)
    return results

def print_classification(classifications, container=st):
    for each in classifications:
        container.markdown(each['sequence'])
        df_classifications = pd.DataFrame([each['scores']],columns=each['labels'])
        container.dataframe(df_classifications)

def app():
    st.title('Dialog Recreation')
    
    summary_container = st.beta_expander("Summary",expanded = True)
    summary_container.header('Summary')
    summary_container.markdown('The app was conceived with the idea of recreating and generate new dialogs for existing games.')
    summary_container.markdown('With this objective, an experiment with the dialogs of Assassin\'s Creed: Odyssey was perform')
    summary_container.markdown('In order to generate a dataset for training the steps followed were:')
    summary_container.markdown('1. Download from [Assassins Creed Fandom Wiki](https://assassinscreed.fandom.com/wiki/Special:Export) from the category "Memories relived using the Animus HR-8.5".')
    summary_container.markdown('2. Keep only text elements from XML.')
    summary_container.markdown('3. Keep only dialog section.')
    summary_container.markdown('4. Parse wikimarkup with [wikitextparser](https://pypi.org/project/wikitextparser/).')
    summary_container.markdown('5. Clean description of dialog\'s context.')
    summary_container.markdown('Due to the small size of the dataset obtained, a transfer learning approach was consider based in a pretrained "Dialog GPT" model.')
    summary_container.markdown('Also a "Zero-Shot Classification" is explored as a multiclass sentiment analysis helper. But this app lends you to modify the target classification and its representation in the following table.')
    session_state = SessionState.get(dialog_recreation_learn=None,
                                     dialog_recreation_tokenizer=None, 
                                     dialog_recreation_classifier=None,
                                     dialog_recreation_prompt=None)
    learn = session_state.dialog_recreation_learn
    tokenizer = session_state.dialog_recreation_tokenizer
    classifier = session_state.dialog_recreation_classifier
    if learn is None:
        with st.spinner('Loading models...It could take more than a 1 min.'):
            learn, tokenizer, classifier = load_models()
            session_state.dialog_recreation_learn = learn
            session_state.dialog_recreation_tokenizer = tokenizer
            session_state.dialog_recreation_classifier = classifier
    
    configurations_container = st.beta_expander("Configurations",expanded = True)
    col_config_generation, col_config_zero = configurations_container.beta_columns(2)
    
    col_config_generation.subheader('Text generation configuration')
    
    #do_sample=col_config_generation.checkbox('Do Sampling', value=True) 
    max_length=col_config_generation.slider('Max generation length', min_value=0, max_value=1000, value=50, step=10)
    top_k=col_config_generation.slider('Top K', min_value=0, max_value=1000, value=50)
    top_p=col_config_generation.slider('Top P', min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    #num_return_sequences=col_config_generation.slider('Number of Samples', min_value=1, max_value=10, value=1)  
    
    col_config_zero.subheader('Zero-shot configuration')
    
    candidate_labels = ['Fun', 'Happiness', 'Love', 'Neutral', 'Sadness', 'Anger', 'Hate']
    candidate_labels_input = col_config_zero.text_area('Classes', str(candidate_labels))
    try:
        candidate_labels = ast.literal_eval(candidate_labels_input)
    except (ValueError,SyntaxError)  as ex:
        col_config_zero.error("Labels requests list format. E.g: ['Fun', 'Happiness']")
        st.stop()
    
    generations_container = st.beta_expander("Generations",expanded = True)
    prompt_space = generations_container.empty()
    if session_state.dialog_recreation_prompt is None:
        prompt = '*Kassandra:'
    else:
        prompt = session_state.dialog_recreation_prompt
        
    generate_button = generations_container.button('Generate')
   
    if generate_button:
        with st.spinner('Generating dialog and stats...'):
            prediciton = get_preds(prompt, learn, tokenizer,
                                   max_length=len(prompt)+max_length, top_k=top_k,
                                   top_p=top_p, num_return_sequences=1, 
                                   candidate_labels = candidate_labels )
            prompt = prediciton
            session_state.dialog_recreation_prompt = prediciton
            classifications = get_classification(prompt, classifier, candidate_labels)
            print_classification(classifications, generations_container)

    session_state.dialog_recreation_prompt = prompt_space.text_area('Input and results', prompt )

    # prompt = "*Kassandra: "
    # prompt_ids = tokenizer.encode(prompt)
    # inp = tensor(prompt_ids)[None]
    # preds = learn.model.generate(inp,
    #     do_sample=True, 
    #     max_length=50, 
    #     top_k=50, 
    #     top_p=0.95, 
    #     num_return_sequences=3 )
    #st.write(tokenizer.decode(preds[0].cpu().numpy()))
    #for i, sample_output in enumerate(preds):
    #    st.write("{}: {}".format(i, tokenizer.decode(sample_output.cpu().numpy(), skip_special_tokens=True)))
    # for i, sample_output in enumerate(preds):
    #     lines = tokenizer.decode(sample_output.cpu().numpy(), skip_special_tokens=True).splitlines()
    #     st.write("{}: {}".format(i, '-'*100))
    #     for line in lines:
    #         line = line.split(':')
    #         if len(line)<2: continue
    #         st.write(classifier(line[1], candidate_labels, multi_class=True))