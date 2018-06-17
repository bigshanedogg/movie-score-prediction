import numpy as np
import pickle
import re
import gensim
import cohesion_probability as tool

####parameter setting

tokenizer = "cohesion"
feature_size = int(input("feature size : "))
#feature_size = 100
min_count = 10
remove_pattern = re.compile('[^ ㄱ-ㅣ가-힣0-9a-zA-Z]+')  # 한글,숫자,영어 제외한 문자


####review data learning
output_path = "pos_data/review_pos/"
review_pos = []
with open(output_path+"review_"+ tokenizer + "_pos.txt", mode="rb") as fp :
        review_pos = pickle.load(fp)

model = gensim.models.Word2Vec(review_pos, size=feature_size, window=5, min_count=min_count, workers=5)
print("#" * 40)
print(tokenizer, "review data learning completed......")
#print("total vocab size:", len(model.wv.vocab))
print("#" * 40)
'''
print("Saving word2vec model......")
model_name = "./w2v_model/review_w2v" + "_" + str(feature_size)
with open(model_name, mode="wb") as fp:
    model.save(fp)
'''

####wiki data learning
batch_len = 90
output_path = "pos_data/" + tokenizer + "_pos/"
wiki_pos = []

#loading wiki_pos from 1 to 90 and batch training
for i in range(1,batch_len+1) :
    print("wiki_pos_"+str(i)+" loading......", str(i/90*100)[0:5]+"%")
    with open(output_path+"cohesion_pos"+str(i)+".txt", mode="rb") as fp :
        wiki_pos = pickle.load(fp)
        model.build_vocab(wiki_pos, update=True)
        model.train(wiki_pos, total_examples=model.corpus_count, epochs=model.iter)


print("#"*40)
print(tokenizer, "wiki data learning completed......")
print(tokenizer, "total vocab size:",len(model.wv.vocab))
print("#"*40)

#saving word2vec model
print("Saving word2vec model......")
model_name = "./w2v_model/" + tokenizer + "_w2v" + "_" + str(feature_size)
with open(model_name, mode="wb") as fp:
    model.save(fp)