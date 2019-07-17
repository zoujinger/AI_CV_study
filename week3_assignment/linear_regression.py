# not use "Too Many For Loops", especially when doing calculations.
# Write the code in "Python's way".

import numpy as np
import random

def inference(w, b, x):
    pred_y = w * np.asarray(x)  + b
    return pred_y

def eval_loss(w, b, x, gt_y):    # ground truth
    loss = 0.5*(inference(w, b, x) - gt_y)**2
    avg_loss = sum(loss)
    avg_loss /= len(x)
    return avg_loss

def gradient(pre_y, gt_y, x):
    diff  =pre_y - gt_y
    dw = diff * x
    db = diff
    return dw, db

def cal_step_gradient(batch_x, batch_gt_y, w, b, lr):
    batch_size = len(batch_x)
    pred_y = inference(w, b, batch_x)
    dw, db = gradient(pred_y, batch_gt_y, batch_x)
    avg_dw = sum(dw) / batch_size
    avg_db = sum(db) / batch_size
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b

def train(x, gt_y, batch_size, lr, max_iter):
    w, b = 0.0, 0.0
    num = len(x)
    for i in range(max_iter):
        batch_idxs = np.random.choice(num, batch_size)
        batch_x = [x[j] for j in batch_idxs]
        batch_y = [gt_y[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss: {0}'.format(eval_loss(w, b, x, gt_y)))

def gen_sample_data():
    w = random.randint(0,9) + random.random()
    b = random.randint(0,5) + random.random()
    print('gen_w. gen_b:', w, b)
    num_samples = 100
    x = np.random.randint(0, 20, num_samples)
    x = x + random.random()
    y = w*x + b + random.random()
    return x, y, w, b

def run():
    x, y, w, b = gen_sample_data()
    lr = 0.01
    max_iter = 1000
    train(x, y, 50, lr, max_iter)

if __name__=='__main__':
    run()