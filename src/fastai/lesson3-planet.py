from fastai.vision import *

path = "../datasets/planet/"
df = pd.read_csv(path + "train_v2.csv")
print(df.head())

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

np.random.seed(42)
src = (ImageList.from_csv(path, 'train_small.csv', folder='train-jpg', suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))

data = (src.transform(tfms, size=128).databunch().normalize(imagenet_stats))

# show images
# data.show_batch(rows=3, figsize=(12,9))

arch = models.resnet18
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=[acc_02, f_score])

lr = 0.1
learn.fit_one_cycle(40, slice(lr))
