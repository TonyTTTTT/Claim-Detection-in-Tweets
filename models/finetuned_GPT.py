import openai


class FineTunedGPT:
    def __init__(self):
        openai.api_key = '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'

    def get_response(self, prompt):
        res = openai.Completion.create(
          model="davinci:ft-personal:clef2022-1a-2023-03-27-19-32-15",
          prompt=prompt + '\ncheckworthy:'
        )
        print('chat input: {}'.format((prompt + '\ncheckworthy:')))
        print('chat response: {}'.format(res.choices[0].text))

        return res.choices[0].text


if __name__ == '__main__':
    prompt = "NEW: Justice Sonia Sotomayor said during oral arguments today that “we have over 100,000 children, which we’ve never had before, in serious condition, and many on ventilators” due to the coronavirus. That's False. https://t.co/9itoVd1s1L https://t.co/zX8Nf6Bx8r"
    chatgpt = FineTunedGPT()
    res = chatgpt.get_response(prompt)
    pass
