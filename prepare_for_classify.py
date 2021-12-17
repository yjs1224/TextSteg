import os
import jsonlines
import csv
from sklearn.model_selection import train_test_split


def sample_for_classification(cover_file, stego_file, out_dir, max_num=10000):
    labels = []
    texts = []
    covers = []
    stegos = []
    os.makedirs(out_dir,exist_ok=True)
    with open(cover_file,"r",encoding="utf-8") as f:
        counter = 0
        for cover in jsonlines.Reader(f):
            if counter >= max_num:
                break
            texts.append(" ".join(cover["cover"].split(" ")[1:-1]))
            covers.append(" ".join(cover["cover"].split(" ")[1:-1]))
            labels.append(0)
            counter += 1


    with open(stego_file,"r",encoding="utf-8") as f:
        counter = 0
        for stego in jsonlines.Reader(f):
            if counter >= max_num:
                break
            texts.append(" ".join(stego["stego"].split(" ")[1:-1]))
            stegos.append(" ".join(stego["stego"].split(" ")[1:-1]))
            labels.append(1)
            counter += 1


    with open(os.path.join(out_dir, "cover.txt"), "w",encoding="utf-8") as f:
        f.write("\n".join(covers))

    with open(os.path.join(out_dir, "stego.txt"), "w",encoding="utf-8") as f:
        f.write("\n".join(stegos))


    def write2file(X,Y, filename):
        datas = []
        i = 0
        with open(filename, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text","label"])
            for x,y in zip(X,Y):
                writer.writerow([x,y])

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, train_size=0.8)
    val_texts,test_texts, val_labels, test_labels = train_test_split(val_texts,val_labels, test_size=0.5)
    write2file(train_texts,train_labels, os.path.join(out_dir,"train.csv"))
    write2file(val_texts,val_labels, os.path.join(out_dir,"val.csv"))
    write2file(test_texts, test_labels,os.path.join(out_dir,"test.csv"))


def sample_for_doc_classification(cover_file, stego_file, out_dir, max_num=1000, doc_num=1000):
    labels = []
    texts = []
    os.makedirs(out_dir,exist_ok=True)
    with open(cover_file,"r",encoding="utf-8") as f:
        doc_counter = 0
        sentence_counter = 0
        doc = []
        for cover in jsonlines.Reader(f):
            doc.append(" ".join(cover["cover"].split(" ")[1:-1]))
            if doc_counter >= doc_num:
                break
            if (sentence_counter+1) % max_num ==  0:
                texts.append(" <NEWSENTENCE> ".join(doc))
                labels.append(0)
                doc = []
                doc_counter += 1
            sentence_counter += 1

    with open(stego_file, "r", encoding="utf-8") as f:
        doc_counter = 0
        sentence_counter = 0
        doc = []
        for stego in jsonlines.Reader(f):
            doc.append(" ".join(stego["stego"].split(" ")[1:-1]))
            if doc_counter >= doc_num:
                break
            if (sentence_counter+1) % max_num ==  0:
                texts.append(" <NEWSENTENCE> ".join(doc))
                labels.append(1)
                doc = []
                doc_counter += 1
            sentence_counter += 1



    def write2file(X,Y, filename):
        datas = []
        i = 0
        with open(filename, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text","label"])
            for x,y in zip(X,Y):
                writer.writerow([x,y])


    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, train_size=0.8)
    val_texts,test_texts, val_labels, test_labels = train_test_split(val_texts,val_labels, test_size=0.5)
    write2file(train_texts,train_labels, os.path.join(out_dir,"train.csv"))
    write2file(val_texts,val_labels, os.path.join(out_dir,"val.csv"))
    write2file(test_texts, test_labels,os.path.join(out_dir,"test.csv"))


if __name__ == '__main__':
    cover_file = "decoding-ac/news/covers-decoding.jsonl"
    stego_file = "generation/encoding/1124-news-ac/stegos-encoding.jsonl"
    sample_for_classification(cover_file,stego_file,"classfication-data/1124-news-ori",max_num=5000)

    stego_file = "generation/encoding/1124-news-ac-oov/stegos-encoding.jsonl"
    sample_for_classification(cover_file, stego_file, "classfication-data/1124-news-1117-2",max_num=5000)

    cover_file = "decoding-ac/movie/covers-decoding.jsonl"
    stego_file = "generation/encoding/1124-movie-ac/stegos-encoding.jsonl"
    sample_for_classification(cover_file,stego_file,"classfication-data/1124-movie-ori",max_num=5000)

    stego_file = "generation/encoding/1124-movie-ac-oov/stegos-encoding.jsonl"
    sample_for_classification(cover_file, stego_file, "classfication-data/1124-movie-1117-2",max_num=5000)

    cover_file = "decoding-ac/tweet/covers-decoding.jsonl"
    stego_file = "generation/encoding/1124-tweet-ac/stegos-encoding.jsonl"
    sample_for_classification(cover_file,stego_file,"classfication-data/1124-tweet-ori",max_num=5000)

    stego_file = "generation/encoding/1124-tweet-ac-oov/stegos-encoding.jsonl"
    sample_for_classification(cover_file, stego_file, "classfication-data/1124-tweet-1117-2",max_num=5000)


    # cover_file = "decoding-ac/movie/covers-decoding.jsonl"
    # stego_file = "encoding-ac/movie/stegos-encoding.jsonl"
    # sample_for_doc_classification(cover_file,stego_file,"doc-classfication-data/movie-ori")
    #
    # stego_file = "encoding-ac/movie/1117-2-stegos-encoding.jsonl"
    # sample_for_doc_classification(cover_file, stego_file, "doc-classfication-data/movie-1117-2")