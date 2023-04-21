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
            temperature=0,
        )
        print('GPT input: {}'.format((messages)))
        print('GPT input #words: {}'.format(len(messages[-1]['content'].split())))
        print('GPT response: {}'.format(res.choices[0].message.content.strip()))
        print('GPT response #words: {}'.format(len(res.choices[0].message.content.strip().split())))

        return res.choices[0].message.content.strip()


if __name__ == '__main__':
    content = "India's gift of 100,000 COVID-19 vaccines arrived Barbados earlier today. This was a very special moment for all Barbadians and I want to thank Prime Minister Modi for his quick, decisive, and magnanimous action in allowing us to be the beneficiary of these vaccines. https://t.co/cSCb40c2mt"
    messages_summary = [
        # {"role": "system", "content": "Please provide a brief summary of the article in no more than 20 words."},
        {"role": "user", "content": content+"\nsummarize:"},
    ]
    messages_explain = [
        # {"role": "system", "content": "Can you explain the following article in detail? Please aim at around 100 words."},
        {"role": "user", "content": content+"\nexplain:"}
    ]
    messages_simplify = [
        {"role": "user", "content": content+"\nsimplify:"}
    ]
    messages_extract = [
        {"role": "system", "content": "What is the main argument or point being made in the statement?"},
        {"role": "user", "content": content}
    ]
    messages_rewrite = [
        # {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to understand?"},
        {"role": "user", "content": content + '\nrewrite:'}
    ]
    chatgpt = ChatGPT()
    res_summary = chatgpt.get_response(messages_summary)
    # res_explain = chatgpt.get_response(messages_explain)
    # res_simplify = chatgpt.get_response(messages_simplify)
    # res_extract = chatgpt.get_response(messages_extract)
    # messages_rewrite = chatgpt.get_response(messages_rewrite)

    # res_split = res_extract.split()