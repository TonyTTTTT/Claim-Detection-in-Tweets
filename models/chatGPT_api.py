# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class ChatGPT:
    def __init__(self):
        openai.api_key = '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'

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_response(self, messages):
        res = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=messages
        )
        print('chat input: {}'.format((messages)))
        print('chat response: {}'.format(res.choices[0].message.content.strip()))

        return res.choices[0].message.content.strip()


if __name__ == '__main__':
    content = "Two weekâ   s sick pay : ð   ¬ ð   § UK Â £ 188.50 What youâ   d get if you lived in : ð   ¦ ð   ¹ Austria Â £ 574.70 ð   ð   ªGermany Â £ 574.70 ð   ¸ ð   ªSweden Â £ 459.76 ð   ³ ð   ± Netherlands Â £ 402.29 ð   ªð   ¸ Spain Â £ 241.37 Statutory sick pay in the UK isnâ   t enough to live on . RT if you want decent #SickPayForAll #Coronavirus #Ridge #Marr"
    messages_summary = [
        # {"role": "system", "content": "content summarizer"},
        {"role": "user", "content": content+"\nsummary:"},
    ]
    messages_explain = [
        {"role": "user", "content": content+"\nexplain:"}
    ]
    messages_simplify = [
        {"role": "user", "content": content+"\nsimplify:"}
    ]
    chatgpt = ChatGPT()
    res_summary = chatgpt.get_response(messages_summary)
    res_explain = chatgpt.get_response(messages_explain)
    res_simplify = chatgpt.get_response(messages_simplify)