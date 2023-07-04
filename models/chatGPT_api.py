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
            # max_tokens=1997,
            # top_p=0.01,
        )
        print('=============================')
        print('GPT input:\n{}'.format((messages)))
        print('GPT input #words: {}'.format(len(messages[-1]['content'].split())))
        print('GPT response:\n{}'.format(res.choices[0].message.content.strip()))
        print('GPT response #words: {}'.format(len(res.choices[0].message.content.strip().split())))
        print('=============================')


        return res.choices[0].message.content.strip()


if __name__ == '__main__':
    tweet1 = "@PrisonPlanet @ezralevant those F3cKTARDS have been eating #BATSOUP and other CRAZY SH!T for centuries. if this #COVID19 #CoronaVIRUS thing happened from that it would have happened by now. this is a fooking #BIOWeapon created in a #CHINA LAB #Agenda21 #Agenda2030= #CHINAISASSHO #UN https://t.co/gSNEWit4EA"
    tweet2 = "Breaking: obscure law requires Sen confirmation for WH aide like Bannon to serve on NSC. 50 U.S. Code § 3021 https://t.co/1sRQEnP3CY — Jonathan Alter (@jonathanalter) January 31, 2017"
    tweet3 = "All I'm saying is meth will cure Coronavirus"
    tweet4 = "Rather than fighting #coronavirus , May be you could use @Lysol to make #TheRealDonaldTrump go away"
    tweet5 = "\"@Nigel_Farage #Coronavirus was #sars virus! #china synthesized it in a lab to be more deadly, more contagious! Using as bio-weapon worldwide now! To weaken other economies globally!\n#Covid_19\""
    tweet6 = "had someone on sc for 2 days before i deleted her ass. she had to have been on some stupid shit. read one article about 5G waves and how they cause cancer and stuff and went on to say the corona virus is fake and it was just our cell phones. is u stupid"
    tweet7 = "So the cure for the virus is to inject ourselves with disinfectant?! Thank you @realDonaldTrump for saving the world..again! #TrumpPressConference #COVID19 #StayAtHome https://t.co/8atw9T9C2I"
    tweet8 = "Hey what disinfectant should I inject myself with? Bleach? Ammonia? Vinegar? h2oo?  #COVID19 @realDonaldTrump @POTUS @WhiteHouse"
    tweet9 = "RT @teaxtarot: It’s 2020 and there’s still no cure for liking men and u expect a corona vaccine? Be realistic."
    tweet10 = "@WelshGasDoc Studies have found that a strong dose of arsenic prevents death from #COVID19 and that’s science fact @POTUS"
    tweet11 = "noooooo whatttt you have corona but your ass its so fat wtffffff this is so unfair"
    tweet12 = "Corona is a black light and America is a cum-stained hotel room"
    tweet13 = "lmaooo corona virus in memphis black people going crazy"
    tweet14 = "Dr. Birx realizing that @realDonaldTrump IS a #StableGenius ...AND if you drink #Clorox or #Lysol you most likely won’t die of #COVID19 https://t.co/bcctoEfwzK"
    tweet15 = "for those puzzled about why chloroquine + #covid19 and when to use, it's not clear. in vitro anti-viral activity for prophylaxis. immune-modulating activity (as in rheumatologic dz) to prevent cytokine storm in pneumonia treatment? a good resource: https://t.co/udPSYPRAK0"

    content = tweet9
    messages_normalize = [
        {"role": "user", "content": content + "\nnormalize:"},
    ]
    messages_normalize_v2 = [
        {"role": "user", "content": "Please help me fix the grammar:" + content},
    ]
    messages_normalize_v3 = [
        {"role": "user", "content": content + "\nfix the grammar:"},
    ]
    messages_normalize_v4 = [
        {"role": "user", "content": content + "\nPlease help me fix the grammar, if you can't, just return original input:"},
    ]
    messages_summary = [
        # {"role": "system", "content": "Please provide a brief summary of the article in no more than 20 words."},
        {"role": "user", "content": content+"\nsummarize:"},
    ]
    messages_explain = [
        # {"role": "system", "content": "Can you explain the following article in detail? Please aim at around 100 words."},
        {"role": "user", "content": content+'\nexplain:'}
    ]
    messages_explain_v2 = [
        # {"role": "system", "content": "Can you explain the following article in detail? Please aim at around 100 words."},
        {"role": "user", "content": content+'\nexplain it in simpler language:'}
    ]
    messages_simplify = [
        {"role": "user", "content": content+"\nsimplify:"}
    ]
    messages_clarify = [
        {"role": "user", "content": content+"\nclarify:"}
    ]
    messages_extract = [
        {"role": "system", "content": "What is the main argument or point being made in the following tweet?"},
        {"role": "user", "content": content}
    ]
    messages_extract_v2 = [
        # {"role": "system", "content": "You are helping me process a tweet."},
        {"role": "user", "content": "What is the main argument or point being made in the following tweet?" + content}
    ]
    messages_extract_v3 = [
        {"role": "system", "content": "You are helping me process a tweet."},
        {"role": "user", "content": "What is the main argument or point being made in the following tweet?" + content}
    ]
    messages_rewrite = [
        # {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to understand?"},
        {"role": "user", "content": content + "\nrewrite:"}
    ]
    messages_rewrite_v2 = [
        # {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to understand?"},
        {"role": "user", "content": content + "\nrewrite the tweet in same pronoun as original content, be sure not to over interpretation."}
    ]
    messages_rewrite_v3 = [
        # {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to understand?"},
        {"role": "user",
         "content": "Please rewrite the following tweet in a way that makes it clearer and more understandable, without adding any extra information or interpretation:" + content}
    ]
    messages_rewrite_v4 = [
        # {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to understand?"},
        {"role": "user",
         "content": "Please correct any grammatical errors in the provided content. Please also remove taggings, hashtags and informal words.\n" + content}
    ]
    messages_rewrite_v5 = [
        # {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to understand?"},
        {"role": "user",
         "content": "Please follow the guidelines to revise the cotent. 1. Correct any grammatical errors. 2. Refrain from adding extra content or over-interpreting the information.\n" + content}
    ]
    messages_rephrase = [
        # {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to understand?"},
        {"role": "user", "content": content + "\nrephrase:"}
    ]
    messages_rephrase_v2 = [
        # {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to understand?"},
        {"role": "user", "content": content + "\nrephrase the tweet in same pronoun as original content, be sure not to over interpretation."}
    ]

    article1 = "通告各位好友，我兒已確定:你若有辦敬老卡每人每個月乘坐交通工具補助480元沒用完，"\
                "可在月底前去便利商店買東西把他用完以免浪費。今天才知道，平白浪費不少!"
    article2 = '媽媽在醫院支援武漢前線醫護發回的消息，這段期間切記不要穿帶毛領或是絨線的衣服外套，較容易吸附病毒，請廣而告之。'
    article3 = '有在煮飯的請注意 菌類不能和茄子一起吃，各位注意到了嗎？今年各種蘑菇特別便宜。這是當醫生的同學轉發的，請注意！轉發「 緊急通知：∵醫大已經死17人，友情提醒:最近醫院急診的患者比較多，大都是蘑菇中毒， 今年蘑菇豐收，蘑菇可以和小白菜一起炒,但不能和茄子一起吃，會中毒，在水焯蘑菇的時候放大蒜，如果大蒜變色了，就有毒，不可食用。而且蘑菇和小米、大黃米千萬不要同吃，會產生一種毒素，醫院治不好，後果很嚴重。'
    article4 = '有在唸大專院校的孩子嗎？昨天教育部針對此次疫情搞了一個紓困措施，只要你的小孩仍就讀大專院校的話，請備妥個人身分證影本、戶口名簿影本、學生自己的銀行帳戶影本，這兩天拿到學務處去申請紓困，有申請就會有（一次3000元，連發三個月共9000元）。不管有沒有受到疫情影響，申請了就有喔！'
    article5 = '溫哥華回來投票的朋友傳來：請各位告訴你們的親朋好友，我剛剛接到我在桃園的弟弟的Line，告訴我今天才收到投票通知，我們過去十幾年都固定在一個國小投票，今年改在一個完全不一樣的地方，另外不可以帶手機，如果是國外回去的一定要帶台灣護照，以前從來沒有這個規定，請大家一定要非常的注意。如果沒有注意投票的地方，臨時發現時恐怕會卡在來回的交通之中而錯過投票。為什麼國外回來的僑胞投票要帶䕶照去投票？是真的還是假新聞？葉元之議員可否幫大家查證一下。'

    messages_CDDTC = [
        {"role": "system", "content": "想像你是一個嚴謹的事實查核人員"},
        {"role": "user", "content": "請將下面訊息中\"可以查核\"且\"值得查核\"的部分挑選出來，並以文中的句子條列:\n" + article1}
    ]
    chatgpt = ChatGPT()

    # messages_rewrite = chatgpt.get_response(messages_rewrite)
    # messages_rewrite_v2 = chatgpt.get_response(messages_rewrite_v2)
    # messages_rewrite_v3 = chatgpt.get_response(messages_rewrite_v3)
    messages_rewrite_v4 = chatgpt.get_response(messages_rewrite_v4)
    # messages_rewrite_v5 = chatgpt.get_response(messages_rewrite_v5)

    # res_explain = chatgpt.get_response(messages_explain)
    # res_explain_v2 = chatgpt.get_response(messages_explain_v2)

    # res_summary = chatgpt.get_response(messages_summary)

    # res_simplify = chatgpt.get_response(messages_simplify)

    # res_clarify = chatgpt.get_response(messages_clarify)

    # res_rephrase = chatgpt.get_response(messages_rephrase)
    # res_rephrase_v2 = chatgpt.get_response(messages_rephrase_v2)

    # res_normalize = chatgpt.get_response(messages_normalize)
    # res_normalize_v2 = chatgpt.get_response(messages_normalize_v2)
    # res_normalize_v3 = chatgpt.get_response(messages_normalize_v3)
    # res_normalize_v4 = chatgpt.get_response(messages_normalize_v4)


    # res_extract = chatgpt.get_response(messages_extract)
    # res_extract_v2 = chatgpt.get_response(messages_extract_v2)
    # res_extract_v3 = chatgpt.get_response(messages_extract_v3)


    # messages_CDDTC = chatgpt.get_response(messages_CDDTC)


    # res_split = res_extract.split()

# openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#         {"role": "system", "content": "想像你是一個嚴謹的事實查核人員"},
#         {"role": "user", "content": "請將下面訊息中\"可以查核\"的部分挑選出來，並以文中的句子條列:\n"
#                                     "通告各位好友，我兒已確定:你若有辦敬老卡每人每個月乘坐交通工具補助480元沒用完，"
#                                     "可在月底前去便利商店買東西把他用完以免浪費。今天才知道，平白浪費不少!"},
#     ]
# )