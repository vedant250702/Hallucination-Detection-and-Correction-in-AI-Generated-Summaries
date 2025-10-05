from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import torch
import torch
import xgboost as xgb


class HallucinationPipeline:
    def __init__(self,nli_model_name,device,llm_model=None):
        # NLI Models Setting
        self.tokenizer=AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model=AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
        self.labels=["CONTRADICTION","NEUTRAL","ENTAILMENT"]
        self.device=device
        self.arbiter=False

        # Large NLI model
        self.roberta_large=BatchNLI("roberta-large-mnli",device=device)
        self.roberta_large_ynie=BatchNLI("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",device=device)

        # Correction models
        self.llm=llm_model

        # Arbiter Model
        self.arbiter_model=xgb.XGBClassifier()
        self.arbiter_model.load_model("./Models/ArbiterModel/arbiter_xgboost_model.json")

        # Storing the results of the predicted models
        self.detection_predicted_labels=[]
        self.detection_predicted_labels_after_correction=[]
        self.corrected_summary=[]


    # -------------------------------------------------------------
    # | FUNCTION 1 :  Detecting the hallucinated parts using NLI. |
    # -------------------------------------------------------------
    def Detection(self,premise,hypothesis):
        inputs=self.tokenizer(premise,hypothesis,return_tensors="pt",truncation=True,padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            logit=self.nli_model(**inputs).logits
            prob=F.softmax(logit,dim=-1)
        prediction=torch.argmax(prob,dim=-1)
        entailment_score=prob[:,2]
        contra_score=prob[:,0]
        return prediction.cpu().tolist(),entailment_score.cpu().tolist(), contra_score.cpu().tolist()
    


    # ----------------------------------------------------------------------
    # | FUNCTION 2 : This involves using the LLM to correct the summaries. |
    # ----------------------------------------------------------------------
    def Correction(self,premise,prompt):
        correction=self.llm.correct(premise,prompt)
        return correction


    # ---------------------------------------------------------------------------
    # | FUNCTION 3 : This involves adding the Tag to the summary  <xx>...</xx>  |
    # ---------------------------------------------------------------------------
    def addTags(self,sentences_list,sent_predicted_labels,size):
        for i in range(size):
            if sent_predicted_labels[i]==0:
                sentences_list[i]=f"<xx> {sentences_list[i]} </xx>"

        prompt_ready_summary=" ".join(sentences_list)
        return prompt_ready_summary


    #  ------------------------------------------------------------------------------------------------------
    #  | FUNCTION 4 : This function helps to chunk the article, because max token for base nli model is 256.| 
    #  ------------------------------------------------------------------------------------------------------
    def chunk_article(self,article,chunk_size=240,stride=128):
        input_ids=self.tokenizer.encode(article)
        chunks=[]
        for i in range(0,len(input_ids),stride):
            new_token=input_ids[i:i+chunk_size]
            decoded_article=self.tokenizer.decode(new_token)
            chunks.append(decoded_article)
        return chunks



    #  -------------------------------------------------------------------------------------------
    #  | FUNCTION 5 : Decider neural network which combines the scores and give the final result |
    #  -------------------------------------------------------------------------------------------
    def arbiterModel(self,data):
        result=self.arbiter_model.predict_proba(data)
        sent_pred_label=np.argmax(result,axis=1)
        sent_pred_label=np.where(sent_pred_label==1,2,0)
        return sent_pred_label,result[::,0],result[::,1]

    #  --------------------------------------------------
    #  | FUNCTION 6 : Integrated all the above methods. |
    #  --------------------------------------------------
    # Input : DataFrame(Premise, Hypothesis, labels[Optional])
    def process(self,df,correct_the_summary=False,arbiter=False):

        # Empty previous outputs.
        self.arbiter=arbiter
        self.detection_predicted_labels=[]
        self.detection_predicted_labels_after_correction=[]
        self.corrected_summary=[]
        sentence_predicted_labels=[]
        summary_factual_score=[]
        summary_contradiction_score=[]
        all_features=[]
        count=1
 

        # Starting the Pipeline of Detection and Correction
        for premise,hypo in np.array(df):

        # | Step 1 |: Splitting and detecting each sentence.
            splitting_sentences= sent_tokenize(hypo)
            size_of_sentence=len(splitting_sentences)
            chunks=self.chunk_article(premise,chunk_size=384,stride=256)
            chunk_sent_pred=[]
            chunk_sent_score=[]
            chunk_contra_score=[]

            chunk_roberta_large_entail=[]
            chunk_roberta_large_neutral=[]
            chunk_roberta_large_contra=[]

            chunk_roberta_large_ynie_contra=[]
            chunk_roberta_large_ynie_neutral=[]
            chunk_roberta_large_ynie_entail=[]

            for article_part in chunks:
                premises=[article_part]*size_of_sentence

                # NLI Model output.
                sent_predicted_labels,sent_predicted_scores, sent_contra_scores= self.Detection(premises, splitting_sentences)

                chunk_sent_pred.append(sent_predicted_labels)
                chunk_sent_score.append(sent_predicted_scores)
                chunk_contra_score.append(sent_contra_scores)
                
                # Other metrics score
                if arbiter:
                    # rouge,entity_overlap,sbert,tfidf=self.metrics.calculate_all(premises,splitting_sentences)
                    contra,neutral,entail=self.roberta_large.predict(premises,splitting_sentences)
                    chunk_roberta_large_contra.append(contra)
                    chunk_roberta_large_neutral.append(neutral)
                    chunk_roberta_large_entail.append(entail)

                    entail,neutral,contra=self.roberta_large_ynie.predict(premises,splitting_sentences)
                    chunk_roberta_large_ynie_contra.append(contra)
                    chunk_roberta_large_ynie_neutral.append(neutral)
                    chunk_roberta_large_ynie_entail.append(entail)




            # Stacking scores of all chunks. [Columns: Chunks, Row:summary_sentences]
            chunk_sent_pred=np.stack(chunk_sent_pred,axis=1)
            chunk_sent_score=np.stack(chunk_sent_score,axis=1)
            chunk_contra_score=np.stack(chunk_contra_score,axis=1)

            sent_predicted_labels=np.max(chunk_sent_pred,axis=1)
            factual_score=np.max(chunk_sent_score,axis=1)
            contra_score=np.min(chunk_contra_score,axis=1)



            # Taking the score which gives more information.
            if arbiter:
                chunk_roberta_large_entail = np.stack(chunk_roberta_large_entail, axis=1)
                chunk_roberta_large_neutral = np.stack(chunk_roberta_large_neutral, axis=1)
                chunk_roberta_large_contra = np.stack(chunk_roberta_large_contra, axis=1)

                # DeBERTa
                chunk_roberta_large_ynie_entail = np.stack(chunk_roberta_large_ynie_entail, axis=1)
                chunk_roberta_large_ynie_neutral = np.stack(chunk_roberta_large_ynie_neutral, axis=1)
                chunk_roberta_large_ynie_contra = np.stack(chunk_roberta_large_ynie_contra, axis=1)

                roberta_large_contra=np.min(chunk_roberta_large_contra,axis=1)
                roberta_large_neutral=np.median(chunk_roberta_large_neutral,axis=1)
                roberta_large_entail=np.max(chunk_roberta_large_entail,axis=1)


                roberta_large_ynie_contra=np.min(chunk_roberta_large_ynie_contra,axis=1)
                roberta_large_ynie_neutral=np.median(chunk_roberta_large_ynie_neutral,axis=1)
                roberta_large_ynie_entail=np.max(chunk_roberta_large_ynie_entail,axis=1)

                # COMBINING FEATURES
                features=np.stack([factual_score,contra_score,
                                   roberta_large_contra,roberta_large_neutral,roberta_large_entail,
                                   roberta_large_ynie_contra,roberta_large_ynie_neutral,roberta_large_ynie_entail                                   
                                   ],axis=1)


                # Arbiter Neural Network Output
                sent_predicted_labels,contra_score,factual_score=self.arbiterModel(features)


            summary_factual_score.append(np.mean(factual_score))
            summary_contradiction_score.append(np.mean(contra_score))
          

            # Label whether summary is factually correct or not
            SummaryPrediction=0 if 0 in sent_predicted_labels else 2
            sentence_predicted_labels.append(sent_predicted_labels)

            self.detection_predicted_labels.append(int(SummaryPrediction))

        # | Step 2 |: Correction of the Summary using LLMs, if parameter correct_the_summary=True.
            if correct_the_summary:
                if SummaryPrediction<=1:
                    prompt=self.addTags(splitting_sentences,sent_predicted_labels,size_of_sentence)
                    corrected_result=self.Correction(premise,prompt)
                    self.corrected_summary.append(corrected_result)
                else:
                    self.corrected_summary.append(None)

            print("Detected : ",count)
            count+=1
            
        output={
                "predictions":self.detection_predicted_labels,
                "corrected_summary":self.corrected_summary,
                "sent_predicted":sentence_predicted_labels,
                "factual_score":summary_factual_score,
                "contradiction_score":summary_contradiction_score,
                "features":all_features
                }
        return output
    


class BatchNLI:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3,use_safetensors=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, premises, hypotheses):
        # Tokenize entire batch at once
        inputs = self.tokenizer(premises, hypotheses, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1) 
        contra=probs[::,0].cpu().tolist()
        neutral=probs[::,1].cpu().tolist()
        entail=probs[::,2].cpu().tolist()
        return contra,neutral,entail