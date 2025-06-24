import pickle

with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(type(model))   
print(model)       