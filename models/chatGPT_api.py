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
    content = "@kraekerc @BonitaEdu @CassanoraL @axelbroad @RandomSusla @OttawaDaddy @berylrcohen @mariann6668 @danrosenbergnet @amelialibertuc1 @jdouglaslittle @DaddabboM @josanchez65 @jhengstler @noasbobs @ESL_fairy @TLMarkides @mrfusco @mkbtuc @the_dramamama @ZackTeitel @JBradshaw01 @BBFarhadi @PowerLrn @DrLauraPinto @munakadri @JCasaTodd @RamonaMeharg @rolandvo @heidi_allum @sarahsanders33 @CoachJCummings @CarolCampbell4 @Educhatter @Stephen_Hurley @BCGovNews @uhwuhna @Joe_Sheik @MarionMoynihan @miketamasi @DirFisherTVDSB @PaulSydor @fordnation @Sflecce @cristina_CP24 @ETFOeducators @osstf @liberal_party @ASPphysician Dr. Peter Jüni said that although vaccines are effective and Ontario’s rollout has insulated vulnerable people from the chance of death, their deployment would not be fast enough to prevent a third lockdown. “We need firmer restrictions than before. https://t.co/Vfm5Dzk1vj"
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
        {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to read? Please aim for around 50 words."},
        {"role": "user", "content": content}
    ]
    chatgpt = ChatGPT()
    # res_summary = chatgpt.get_response(messages_summary)
    # res_explain = chatgpt.get_response(messages_explain)
    # res_simplify = chatgpt.get_response(messages_simplify)
    res_extract = chatgpt.get_response(messages_extract)
    res_extract_split = res_extract.split()