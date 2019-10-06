from fastai.tabular import *
from fastai.metrics import *

path = Path('../datasets')
file = path/"dataset-er.csv"

df = pd.read_csv(file)
df = df.drop(columns=["sid","time","r_site_iab_cats","g_event_id"])
df.head()

procs = [FillMissing, Categorify, Normalize]

train = df[0:int(len(df)*0.9)].copy()
test = df[int(len(df)*0.9):len(df)].copy()

label = 'response'
cat_names = [cat for cat in df.columns][:-1]
emb_szs = { cat:15 for cat in cat_names}

test_t = TabularList.from_df(test, cat_names=cat_names, cont_names=[], procs=procs)


data = (TabularList.from_df(train, path='.', cat_names=cat_names, procs=procs)
        .split_by_idx(list(range(0,200)))
        .label_from_df(cols = label)
        .add_test(test_t, label=label)
        .databunch())
print(data.train_ds.cat_names)
print(data.train_ds.cont_names)

(cat_x,cont_x),y = next(iter(data.train_dl))
for o in (cat_x, cont_x, y):
    print(to_np(o[:5]))

f1_score =FBeta(average='macro',beta = 1) #partial(fbeta, thresh=0.2, beta = 1)
# acc_02 = partial(accuracy, thresh=0.2)
learn = tabular_learner(data, layers=[1000, 200, 15], emb_szs=emb_szs, metrics=[accuracy,f1_score],emb_drop=0.1, callback_fns=ShowGraph)

result = learn.predict(test.iloc[50])
print(result)
result = learn.predict(test.iloc[150])
print(result)

#
# Calculate Recall and Precison and Fscore metrics from the test set
#
predictions, *_ = learn.get_preds(DatasetType.Test)
labels = test['response']
submission = pd.DataFrame({'prediction' :predictions[:,1] ,'label': labels})

recall = submission[ submission['label'] == 1 ].mean()['prediction']

print("recall: ", recall)

res = []
for i in range(0,len(train)):
    pred = learn.predict(train.iloc[i])
    cat = pred[1].numpy()
    val = pred[2].numpy()[1]
    res.append((cat,val))

print("a")

arr = pd.DataFrame(res)
arr[arr[1] >= 0.1]