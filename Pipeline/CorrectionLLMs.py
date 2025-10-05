# import google.generativeai as genai
# from llama_cpp import Llama
from openai import OpenAI

# class GeminiLLM:
#     def __init__(self,api_key):
#         genai.configure(api_key=api_key)
#         self.model=genai.GenerativeModel('gemini-1.5-flash')

#     def correct(self,article,summary):
#         prompt=f"""
#             Here is a summary with hallucinated parts marked using <xx> tags.

#             Please correct only the text inside the <xx> tags to make it factually accurate based on the original article. Leave the rest of the summary unchanged and remove the <xx> tags after correction.

#             Return the summary with hallucinated parts fixed and you can remove those <xx></xx> tags. Don't remove that entire sentence.
            
#             Original Article:
#             {article}

#             Summary:
#             {summary}

#         """
#         output=self.model.generate_content(prompt)
#         return output.text.strip()



# class OfflineLLMs:
#     def __init__(self,model_name,max_tokens=50):
#         self.model=Llama(model_name,n_ctx=512,gpu_layers=0)
#         self.max_tokens=max_tokens

#     def correct(self,premise,summary):
#         # prompt=f"""
#         #     Rewrite the following sentence to be factually correct based on the provided information. Output should be Only one single sentence that is the factually correct part of hallucinated sentence.
#         #     Its a request give only the output don't rewrite the article and sentence to correct again.

#         #     Information:
#         #     {premise}

#         #     Sentence to correct:
#         #     {summary}

#         #     Corrected sentence:
#         # """
#         prompt=f"""
#             Please rewrite the correct text only the text inside the <xx>..</xx> tags to make it factually accurate based on the original article. Leave the rest of the summary unchanged and remove the <xx> tags after correction.
#             Return the summary with hallucinated parts fixed. Don't remove that entire sentence. Keep the length of summary same as hallucinated summary.
#             Don't add any explanation of the cause.
            
#             Original Article:
#             {premise}

#             Hallucinated Summary:
#             {summary}

#             Corrected Summary:

#         """

#         output=self.model(prompt,max_tokens=self.max_tokens)
#         return output["choices"][0]["text"]
    
    
#     def create(self,article):
#         prompt=f"""
#             Create the summary of the following article given below. Stick to the article for summary. Only give the summary output.
#             Article:
#             {article}

#             Summary:
#         """

#         output=self.model(prompt,max_tokens=self.max_tokens)
#         return output["choices"][0]["text"]


class DeepseekAPI:
    def __init__(self,api_key):
        self.client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
        )

    def correct(self,premise,summary):
        prompt=f"""
            Here is a summary with hallucinated parts marked using <xx> tags.

            Please correct only the text inside the <xx> tags to make it factually accurate based on the original article. 
            Leave the rest of the summary unchanged and remove the <xx> tags after correction.

            Return the summary with hallucinated parts fixed and you can remove those <xx></xx> tags. 
            Don't remove that entire sentence.
            Output only corrected summary, don't give any title.
            
            Original Article:
            {premise}

            Hallucinated Summary:
            {summary}

            Corrected Summary:
        """
        response = self.client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free", 
        # model="deepseek/deepseek-r1-0528:free" ,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that corrects factual errors in summaries."},
            {"role": "user", "content": f"{prompt}"}

        ]
    )
        return response.choices[0].message.content