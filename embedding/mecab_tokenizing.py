import numpy as np
import re
import pickle
import csv
import cohesion_probability as tool
from konlpy.tag import Mecab
#from konlpy.tag import Kkma

mecab = Mecab()
remove_pattern = re.compile('[^ ㄱ-ㅣ가-힣0-9a-zA-Z]+')  # 한글,숫자,영어 제외한 문자

review_data = []
with open("data_prepared.csv", mode="r", encoding="utf-8") as fp :
    rdr = csv.reader(fp)
    for row in rdr :
        review_data.append(row)
review_data = review_data[1:65001]
x_data = np.array(review_data)[:,1:].flatten()
y_data = list(map(int, np.array(review_data)[:,0]))
x_data = [remove_pattern.sub("", review) for review in x_data]

print("labels:", len(y_data))
with open('labels.csv', 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=" ")
    for i in range(len(y_data)) :
        csv_writer.writerow([i+1,",", y_data[i]])


wiki_data = []
wiki_file_name = "./pos_data/wiki_source/wiki"
for i in range(1, 91) :
    with open(wiki_file_name+str(i)+".txt", mode="rb") as fp :
        temp = pickle.load(fp)
        wiki_data = wiki_data + temp

cohesion_training_data = wiki_data + x_data

#cohesion_training_data = x_data
cohesion = tool.CohesionProbability()
cohesion.train(cohesion_training_data)
cohesiontokenizer = tool.CohesionTokenizer(cohesion)


cohesion_tokenized_reviews = [[token for divided in review.split() for token in cohesiontokenizer.tokenize(divided)] for review in x_data]
print("cohesion_tokenized_reviews:", len(cohesion_tokenized_reviews))
with open('cohesion_tokens.csv', 'w', encoding='utf-8', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=" ")
    for i in range(len(cohesion_tokenized_reviews)) :
        csv_writer.writerow([i+1, ",", " ".join(cohesion_tokenized_reviews[i])])


tokenized_reviews = [[element[0] for word in review for element in mecab.pos(word)] for review in cohesion_tokenized_reviews]
print("tokenized_reviews:", len(tokenized_reviews))
with open('mecab_tokens.csv', 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=" ")
    for i in range(len(tokenized_reviews)) :
        csv_writer.writerow([i+1, ",", " ".join(tokenized_reviews[i])])

tokenized_reviews_tag = [[element[1] for word in review for element in mecab.pos(word)] for review in cohesion_tokenized_reviews]
print("tokenized_reviews_tag:", len(tokenized_reviews_tag))
with open('mecab_tokens_tag.csv', 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=" ")
    for i in range(len(tokenized_reviews_tag)) :
        csv_writer.writerow([i+1, ",", " ".join(tokenized_reviews_tag[i])])


'''
output_path = "pos_data/review_pos/"
with open(output_path+"review_pos.txt", mode="wb") as fp :
    pickle.dump(tokenized_reviews, fp)

print("x_data len: %d, y_data len: %d" %(len(x_data), len(y_data)))
np.savetxt("labels.csv", y_data, fmt="%i", delimiter=",")
'''