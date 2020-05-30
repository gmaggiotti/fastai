from fastai.vision import *
from fastai.metrics import error_rate

batch_size = 64
path = untar_data(URLs.PETS); path

path_annotations = path / 'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)
fnames[:5]


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=50, bs=batch_size
                                   ).normalize(imagenet_stats)

np.random.seed(2)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

print("loading model.")
learn.load('stage-1');

# print("finding LR")
# learn.lr_find()
#
# print("ploting LR")
# learn.recorder.plot()
# learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-4))

interp = ClassificationInterpretation.from_learner(learn)
most_c= interp.most_confused(min_val=5)
for (wrong, good, diff ) in most_c:
    print(wrong,",",good,",",diff)
exit()




# If it doesn't, you can always go back to your previous model.

# In[ ]:


learn.load('stage-1-50');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused(min_val=2)


# ## Other data formats

# In[ ]:


path = untar_data(URLs.MNIST_SAMPLE); path


# In[ ]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)


# In[ ]:


df = pd.read_csv(path/'labels.csv')
df.head()


# In[ ]:


data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))
data.classes


# In[ ]:


data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
data.classes


# In[ ]:


fn_paths = [path/name for name in df['name']]; fn_paths[:2]


# In[ ]:


pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
data.classes


# In[ ]:


data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
        label_func = lambda x: '3' if '/3/' in str(x) else '7')
data.classes


# In[ ]:


labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
labels[:5]


# In[ ]:


data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)
data.classes


# In[ ]:




