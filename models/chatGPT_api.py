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
        print('chat input: {}'.format((messages)))
        print('chat response: {}'.format(res.choices[0].message.content.strip()))

        return res.choices[0].message.content.strip()


if __name__ == '__main__':
    content = "Show me who all the selfish and sick people are after they have been shown countless lives being lost from the new Covid-19 vaccines (hopefully newer CV vaccines are safer), whenever they post 💉 📸 online."
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
        {"role": "system", "content": "Rewrite the following article to be more concise and understandable."},
        {"role": "user", "content": content}
    ]
    chatgpt = ChatGPT()
    # res_summary = chatgpt.get_response(messages_summary)
    # res_explain = chatgpt.get_response(messages_explain)
    # res_simplify = chatgpt.get_response(messages_simplify)
    res_extract = chatgpt.get_response(messages_extract)