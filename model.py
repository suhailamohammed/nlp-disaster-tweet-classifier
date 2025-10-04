from sklearn.linear_model import LogisticRegression

def train_model(x, y):
    model = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs', random_state=42)
    model.fit(x, y)

    return model

def test_model(model, embeddings):
    y_pred = model.predict(embeddings)
    
    return y_pred