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
            messages=messages,
            # temperature=0,
        )
        print('GPT input: {}'.format((messages)))
        print('GPT input #words: {}'.format(len(messages[-1]['content'].split())))
        print('GPT response: {}'.format(res.choices[0].message.content.strip()))
        print('GPT response #words: {}'.format(len(res.choices[0].message.content.strip().split())))

        return res.choices[0].message.content.strip()


if __name__ == '__main__':
    content = "BREAKING: Trump just told the press that a Coronavirus vaccine will be ready in 3-4 months. Seconds later, in the same exact room, an expert tells the press that a vaccine will be ready in 12-18 months at the earliest. THIS IS A PROBLEM! Trump is a train wreck!"
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
    messages_extract = [
        {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to read? Please do not exceed 30 words."},
        {"role": "user", "content": content}
    ]
    chatgpt = ChatGPT()
    # res_summary = chatgpt.get_response(messages_summary)
    # res_explain = chatgpt.get_response(messages_explain)
    # res_simplify = chatgpt.get_response(messages_simplify)
    res_extract = chatgpt.get_response(messages_extract)
    res_extract_split = res_extract.split()