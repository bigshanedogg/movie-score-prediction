import gensim
import pickle
import numpy as np
import csv


def w2v_embedding(tokenizer, feature_size) :
    model_path = "./w2v_model/" + tokenizer + "_w2v"
    model = gensim.models.Word2Vec.load(model_path + "_" + str(feature_size))

    min_count = model.min_count
    feature_size = model.vector_size
    print(tokenizer, "w2v\n\treview min_count: %d, review feature_size: %d" %(min_count, feature_size))


    output_path = "pos_data/review_pos/"
    review_pos = []
    with open(output_path+"review_" + tokenizer + "_pos.txt", mode="rb") as fp :
            review_pos = pickle.load(fp)

    vecs = []
    print("\tvector embedding using " + tokenizer + "_w2v")
    for row in review_pos :
        #단어장에 없을 경우 해당 형태소 제거
        row_vec = [model.wv[morp] for morp in row if morp in model.wv.vocab]

        # 단어장에 없을 경우 랜덤 분포로 임의 추가
        #row_vec = [model.wv[morp] if morp in model.wv.vocab else np.random.normal(0, 1, feature_size) for morp in row]
        vecs.append(row_vec)

    vecs = np.array(vecs)
    vecs_flatten = np.array([np.array(v).flatten() for v in vecs])

    model_name = "review_" + tokenizer + "_" + str(feature_size) + ".csv"
    with open(model_name, mode="w", encoding="utf-8") as fp :
        wr = csv.writer(fp)
        for v in vecs_flatten :
            wr.writerow(v)


def get_vec_from_csv(csv_path, feature_size) :
    vec = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            if len(row) < 1:
                vec.append(np.array([]))
                continue
            temp = np.array([float(v) for v in row[0].split(',')])
            temp = np.reshape(temp, [-1, feature_size])
            vec.append(temp)

    return vec


if __name__=="__main__" :
    #feature_size = [50, 100, 200, 300]
    feature_size = [50, ]
    for i in feature_size :
        w2v_embedding("mecab", i)
        w2v_embedding("cohesion", i)


