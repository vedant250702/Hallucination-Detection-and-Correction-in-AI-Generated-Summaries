from nltk.tokenize import sent_tokenize


# This method is created because it will be required for 
# Detection as well as Correction (Like results after correction)

def detectionProcess(article,summary,pipeline,arbiter):
    result=pipeline.process([[article,summary]],correct_the_summary=False,arbiter=arbiter)
    all_sentences=sent_tokenize(summary)
    print(result)
    summary=pipeline.addTags(all_sentences,result["sent_predicted"][0],len(all_sentences))
    score=str(result["factual_score"][0])
    sentenceLabels=list(result["sent_predicted"][0])
    labelCounts=[sentenceLabels.count(0),sentenceLabels.count(2)]
  
    prompt=f"""
Here is a summary with hallucinated parts marked using <xx> tags.

Please correct only the text inside the <xx> tags to make it factually accurate based on the original article. Leave the rest of the summary unchanged and remove the <xx> tags after correction.

Return the summary with hallucinated parts fixed and you can remove those <xx></xx> tags. Don't remove that entire sentence.
            
Original Article:
    {article}

Summary:
    {summary}

"""
    return {"summary":summary,"score":score,"counts":labelCounts,"copy_prompt":prompt}