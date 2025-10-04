from dataset_loader import load_dataset
from preprocessing import clean_text
from embeddings import tweets_into_embeddings
from model import train_model, test_model


if __name__ == "__main__":
    train_df = load_dataset("train")

    y = train_df.target.values

    train_df = train_df.drop("target", axis=1)

    train_df["cleaned_text"] = train_df.text.apply(clean_text)

    ## converting text into vectors
    texts = train_df.cleaned_text.tolist()
    x = tweets_into_embeddings(texts, embeddingType="bert") #embeddingType: word2vec to use word2vec embedding method

    ## train model
    model = train_model(x, y)

    ##test model
    test_df = load_dataset("test")
    test_df["cleaned_test"] = test_df.text.apply(clean_text)
    test_texts = test_df.cleaned_text.tolist()
    x_test = tweets_into_embeddings(test_texts, embeddingType="bert")
    y_pred = test_model(model, x_test)

    #Sample submission
    sample_submission = load_dataset('sample_submission.csv')
    sample_submission["target"] = y_pred

    sample_submission.to_csv("submission.csv", index=False)
    





