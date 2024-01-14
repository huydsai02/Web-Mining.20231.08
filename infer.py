from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers import BertPreTrainedModel,  BertModel, AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import torch, re
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pickle, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

category_ids = {'negative': 0, 'positive': 1}
ids_to_label = {0 : 'negative', 1 : 'positive'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern       = '@[^\s]+'
tagPattern        = '#[^\s]+'
alphaPattern      = "[^a-zA-Z0-9]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

stopwordlist = set(['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves'] + [w for w in stopwords.words('english')])


class BertCNNForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = False

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(806, config.num_labels)

        self.conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 768), padding=(1,1))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.flat = nn.Flatten()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        # print(outputs['hidden_states'][-1].shape)
        x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in outputs['hidden_states']]), 0), 0, 1)
        # print(x.shape)
        x = self.pool(self.dropout(self.relu(self.conv(self.dropout(x)))))
        logits = self.classifier(self.dropout(self.flat(self.dropout(x))))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return loss, logits
    

class BERTInfer:
    def __init__(self, bert_checkpoint):
        model_name_or_path = bert_checkpoint
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=len(category_ids)
        )
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.HASHTAG_SPEC = 'tagdaynheem'
        self.USER_SPEC = 'userdaynheem'
        self.URL_SPEC = 'urldaynheem'
        self.EMOJI_SPEC = 'emojidaynheem'
        self.max_input_length = 256

    def preprocess_text(self, tweet):
        tweet = tweet.lower()    
            # Replace all URls with 'URL'
        tweet = re.sub(urlPattern, f' {self.URL_SPEC} ' , tweet)
            # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, f" {self.EMOJI_SPEC} " + emojis[emoji])        
            # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern, f' {self.USER_SPEC} ' , tweet)   
        tweet = re.sub(tagPattern, f' {self.HASHTAG_SPEC} ' , tweet)
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " " , tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        return tweet
    
    def infer(self, text):
        model_inputs = self.tokenizer([text], max_length=self.max_input_length, truncation=True, return_tensors='pt')
        for k in model_inputs:
            model_inputs[k] = model_inputs[k].to(device)
        out = torch.argmax(self.model(**model_inputs)['logits'][0]).item()
        return ids_to_label[out]
    
class BERTCNNInfer:
    def __init__(self, bert_checkpoint):
        model_name_or_path = bert_checkpoint
        model = BertCNNForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=len(category_ids)
        )
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.HASHTAG_SPEC = 'HASHTAG'
        self.USER_SPEC = 'USER'
        self.URL_SPEC = 'URL'
        self.EMOJI_SPEC = 'EMOJI'
        self.max_input_length = 64

    def preprocess_text(self, tweet):
        tweet = tweet.lower()    
            # Replace all URls with 'URL'
        tweet = re.sub(urlPattern, f' {self.URL_SPEC} ' , tweet)
            # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, f" {self.EMOJI_SPEC} " + emojis[emoji])        
            # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern, f' {self.USER_SPEC} ' , tweet)   
        tweet = re.sub(tagPattern, f' {self.HASHTAG_SPEC} ' , tweet)
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " " , tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        return tweet
    
    def infer(self, text):
        model_inputs = self.tokenizer([text], max_length=self.max_input_length, truncation=True, pad_to_max_length=True, return_tensors='pt')
        for k in model_inputs:
            model_inputs[k] = model_inputs[k].to(device)
        out = torch.argmax(self.model(**model_inputs)[1][0]).item()
        return ids_to_label[out]
    
class MLInfer:
    def __init__(self, *, vectorise_path, checkpoint_path):
        with open(vectorise_path, 'rb') as f:
            self.vectorise = pickle.load(f)

        with open(checkpoint_path, 'rb') as f:
            self.model = pickle.load(f)

        self.HASHTAG_SPEC = 'HASHTAG'
        self.USER_SPEC = 'USER'
        self.URL_SPEC = 'URL'
        self.EMOJI_SPEC = 'EMOJI'
        self.wordLemm = WordNetLemmatizer()

    def preprocess_text(self, tweet):
        tweet = tweet.lower()    
            # Replace all URls with 'URL'
        tweet = re.sub(urlPattern, f' {self.URL_SPEC} ' , tweet)
            # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, f" {self.EMOJI_SPEC}" + emojis[emoji])        
            # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern, f' {self.USER_SPEC} ' , tweet)   
        tweet = re.sub(tagPattern, f' {self.HASHTAG_SPEC} ' , tweet)
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " " , tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
        tweet = ' '.join([self.wordLemm.lemmatize(word) for word in tweet.split() if len(word) > 1 and word not in stopwordlist])
        return tweet
    
    def infer(self, text):
        data = self.vectorise.transform([text])
        out = self.model.predict(data)[0]
        return ids_to_label[out]


if __name__ == '__main__':

    input_text = 'hello my friend'
    
    bert = BERTInfer('bert')
    print(bert.infer(input_text))

    bertcnn = BERTCNNInfer('bertcnn')
    print(bertcnn.infer(input_text))

    lr = MLInfer(
        vectorise_path='ml/vectoriser-ngram-(1,2).pickle',
        checkpoint_path='ml/LR.pickle'
    )
    print(lr.infer(input_text))

    bnb = MLInfer(
        vectorise_path='ml/vectoriser-ngram-(1,2).pickle',
        checkpoint_path='ml/BNB.pickle'
    )
    print(bnb.infer(input_text))

    svc = MLInfer(
        vectorise_path='ml/vectoriser-ngram-(1,2).pickle',
        checkpoint_path='ml/SVCmodel.pickle'
    )
    print(svc.infer(input_text))
