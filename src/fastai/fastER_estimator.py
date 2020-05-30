from fastai.tabular import *
from fastai.metrics import *

path = Path('datasets/')
file = path/"dataset-er.csv"

df = pd.read_csv(file)
df = df.drop(columns=["sid","time","r_site_iab_cats","g_event_id"])
df.head()

procs = [FillMissing, Categorify, Normalize]
valid_idx = range(int(len(df)*0.9), len(df))


label = 'response'
cat_names = [cat for cat in df.columns][:-1]
emb_szs = { cat:15 for cat in cat_names}

data = TabularDataBunch.from_df(path, df, label, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
print(data.train_ds.cont_names)

(cat_x,cont_x),y = next(iter(data.train_dl))
for o in (cat_x, cont_x, y):
    print(to_np(o[:5]))

f1_score =FBeta(average='macro',beta = 1) #partial(fbeta, thresh=0.2, beta = 1)
learn = tabular_learner(data, layers=[200,100], emb_szs=emb_szs, metrics=[accuracy,f1_score])
learn.fit_one_cycle(6)

result = learn.predict(df.iloc[50])
print(result)
result = learn.predict(df.iloc[150])
print(result)



