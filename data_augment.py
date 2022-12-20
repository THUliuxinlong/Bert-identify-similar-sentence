import pandas as pd
import jionlp as jio

tencent_api = jio.TencentApi(
    [{"project_id": "1315406438",
      "secret_id": "AKIDCWZWOKLOgzNtMrQ3eEqWTL0TjohMmyvg",
      "secret_key": "e7QXQ7bqsiNp471mdLq7Azd3EBaqukNw"}])


baidu_api = jio.BaiduApi(
    [{'appid': '20221127001473225',
      'secretKey': 'jOcVvNNFam4fub1VKHZ8'}], gap_time=0.5)

# print(baidu_api.__doc__)  # 查看接口说明
# print(tencent_api.__doc__)  # 查看接口说明
# print(youdao_api(text))  # 使用接口做单次调用

apis = [baidu_api]#[baidu_api, youdao_api, google_api, tencent_api, xunfei_api]
back_trans = jio.BackTranslation(mt_apis=apis)
# text = '饿了么凌晨发文将推出新功能，用户可选择是否愿意多等外卖员 5 分钟，你愿意多等这 5 分钟吗？'
# print(text)
# result = back_trans(text)
# print(result)

def clean_special_chars(text):
    aug_text = back_trans(text)
    # 如果非空返回翻译，否则返回原文本
    if aug_text:
        return str(aug_text[0])
    else:
        return text

train_df = pd.read_csv('./data/train.csv')
print(train_df.head())
print(train_df.shape)

for i in range(874):
    print('part', i)
    half_train = train_df[i*10:(i+1)*10]
    half_train['query2'] = half_train['query2'].apply(lambda x:clean_special_chars(x))
    half_train.to_csv('./data/back_trans.tsv', sep=',', header=False, index=False, mode='a+')
print('augment finish')
