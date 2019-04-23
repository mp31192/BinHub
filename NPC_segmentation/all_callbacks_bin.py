import keras.callbacks
import numpy as np
import os

class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self,model,path):
        self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        save_path = os.path.join(self.path,'epoch_model.h5')
        self.model.save_weights(save_path,overwrite=True)


#
# class ImageSaver(keras.callbacks.Callback):
#     def __init__(self,r,c,image_savepath):
#         self.x = data
#         self.y = label
#
#     def on_batch_end(self, batch, logs=None):
#         result_imgs = self.model.predict(self.x,batch_size=1,verbose=1)
#         r, c = 2, 2
#         fig, axs = plt.subplots(r, c)
#         cnt = 0
#         for i in range(r):
#             for j in range(c):
#                 if i == 0:
#                     if j == 0:
#                         axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
#                         axs[i, j].axis('off')
#                     elif j == 1:
#                         axs[i, j].imshow(t2_imgs[:, :], cmap='gray')
#                         axs[i, j].axis('off')
#                 elif i == 1:
#                     if j == 0:
#                         axs[i, j].imshow(gen_imgs[cnt, :, :, 1], cmap='gray')
#                         axs[i, j].axis('off')
#                     elif j == 1:
#                         axs[i, j].imshow(label_imgs[:, :], cmap='gray')
#                         axs[i, j].axis('off')
#         fig.savefig(
#             "/media/root/3339482d-9d23-44ee-99a0-85e517217d15/NPC_mri_segmentation_miccai/data/image_show/%d.png" % num)
#         plt.close()