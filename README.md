# Intro
MIT 6.869 class project for scene recognition using an early version(0.11) of **Tensorflow**

This folder contains two of our final models, Res50 and BanRes.

To train, use python3 and run 'res50.py' or 'ban_res2.py'. However, you will need the dataset to run the code.

These two models share the same dataloader called read_batch.py.
But they have different files to define the parameters, which are
called para50.py and ban_para2.py respectively.
