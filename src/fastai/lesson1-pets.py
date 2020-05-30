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

print(data.classes)
len(data.classes),data.c

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(1)

print("End of fitting")
learn.save('stage-1')
exit()



# ## Results

# Let's see what results we have got. 
# 
# We will first see which were the categories that the model most confused with one another. We will try to see if what the model predicted was reasonable or not. In this case the mistakes look reasonable (none of the mistakes seems obviously naive). This is an indicator that our classifier is working correctly. 
# 
# Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily skewed: the model makes the same mistakes over and over again but it rarely confuses other categories. This suggests that it just finds it difficult to distinguish some specific categories between each other; this is normal behaviour.


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)



interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


doc(interp.plot_top_losses)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# ## Unfreezing, fine-tuning, and learning rates

# Since our model is working as we expect it to, we will *unfreeze* our model and train some more.

# unfreeze do not use by default transfer learning.  Allows us to train the model from the scratch


learn.unfreeze()




learn.fit_one_cycle(1)

learn.load('stage-1');



learn.lr_find()
learn.recorder.plot()


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# That's a pretty accurate model!

# ## Training: resnet50

# Now we will train in the same way as before but with one caveat: instead of using resnet34 as our backbone we will use resnet50 (resnet34 is a 34 layer residual network while resnet50 has 50 layers. It will be explained later in the course and you can learn the details in the [resnet paper](https://arxiv.org/pdf/1512.03385.pdf)).
# 
# Basically, resnet50 usually performs better because it is a deeper network with more parameters. Let's see if we can achieve a higher performance here. To help it along, let's us use larger images too, since that way the network can see more detail. We reduce the batch size a bit since otherwise this larger network will require more GPU memory.


data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=batch_size // 2).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet50, metrics=error_rate)



learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(8)


learn.save('stage-1-50')


# It's astonishing that it's possible to recognize pet breeds so accurately! Let's see if full fine-tuning helps:
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))


# If it doesn't, you can always go back to your previous model.
learn.load('stage-1-50');

interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)


# ## Other data formats
path = untar_data(URLs.MNIST_SAMPLE); path

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)


data.show_batch(rows=3, figsize=(5,5))


learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit(2)


df = pd.read_csv(path/'labels.csv')
df.head()


data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)

data.show_batch(rows=3, figsize=(5,5))
data.classes




data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
data.classes


fn_paths = [path/name for name in df['name']]; fn_paths[:2]


pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
data.classes


data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,
        label_func = lambda x: '3' if '/3/' in str(x) else '7')
data.classes


labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]
labels[:5]


data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)
data.classes



