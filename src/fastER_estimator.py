from fastai.tabular import *
from fastai.metrics import error_rate

path = Path('../datasets')
(path/"").ls()

df = pd.read_csv(path/'20190919-to-20190921.csv')
df = df.drop(columns="sid")
df = df.drop(columns="time")
df.head()