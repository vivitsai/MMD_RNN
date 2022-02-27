from snownlp import SnowNLP

text = '美容液是完完全全的水，一点都不浓，眼霜太干了。'

results = SnowNLP(text)


print('输入:', text)
print('情感评分:', results.sentiments)

