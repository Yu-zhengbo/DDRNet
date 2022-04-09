import matplotlib.pyplot as plt
import scipy.signal
import os

logs = os.listdir('./logs')

#[10:15]
losses,val_loss = [],[]
for i in logs:
    try:
        losses.append(float(i[10:15]))
        val_loss.append(float(i[-9:-4]))
    except:
        pass



iters = range(len(losses))
plt.figure()
# plt.plot(iters,losses,'red',linewidth=2,label='train loss')
plt.plot(iters,val_loss,'coral',linewidth=2,label='val f1')

num = 15

# plt.plot(iters, scipy.signal.savgol_filter(losses, num, 3), 'green', linestyle='--', linewidth=2,
#          label='smooth train loss')
plt.plot(iters, scipy.signal.savgol_filter(val_loss, num, 3), '#8B4513', linestyle='--',
         linewidth=2, label='smooth val f1')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()



