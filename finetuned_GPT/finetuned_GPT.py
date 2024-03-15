import openai


class FineTunedGPT:
    def __init__(self):
        with open('../openai_api_key', 'r') as f:
            openai.api_key = f.readlines()[0]

    def get_response(self, prompt):
        res = openai.Completion.create(
            # model="davinci:ft-personal:clef2022-1a-2023-03-27-19-32-15",
            model="davinci:ft-personal:clef2022-1a-ep10-2023-03-28-04-07-27",
            prompt=prompt + '\ncheckworthy:'
        )
        print('chat input: {}'.format((prompt + '\ncheckworthy:')))
        print('chat response: {}'.format(res.choices[0].text))

        return res.choices[0].text


if __name__ == '__main__':
    prompt = "Sorry Dr Fauci and other fearmongers, new study shows vaccines and naturally acquired immunity DO effectively neutralize COVID variants. Good news for everyone but bureaucrats and petty tyrants! Neutralizing Antibodies Against SARS-CoV-2 Variants https://t.co/k4SKSfxLJh"
    chatgpt = FineTunedGPT()
    res = chatgpt.get_response(prompt)
    pass
