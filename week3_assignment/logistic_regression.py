# complete the logistic regression code in "Python's way" as well.
# Tips: It's almost like the linear regression code.
# The only difference is you need to complete a sigmoid function
# and use the result of that as your "new X"
# and also you need to generate your own training data.

import numpy as np
import random

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def inference(w, b, x):
    pred_y = sigmoid(w * np.asarray(x) + b)
    return pred_y

def eval_loss(w, b, x, gt_y):
    pred_y = inference(w, b, x)
    loss = 0.5 * (pred_y - gt_y) ** 2
    avg_loss = sum(loss)/len(gt_y)
    return avg_loss

def gradient(pred_y, gt_y, x):
    diff = (pred_y - gt_y) * d_sigmoid(np.asarray(x))
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

def train(x, y, batch_size, lr, max_iter):
    w, b = 0, 0
    num = len(x)
    for i in range(max_iter):
        batch_idxs = np.random.choice(num, batch_size)
        batch_x = [x[j] for j in batch_idxs]
        batch_y = [y[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0},b:{1}'.format(w,b))
        print('loss:', eval_loss(w, b, x, y))

def gen_sample_data():
    w = np.random.randint(0, 10) + random.random()
    b = np.random.randint(0, 5)  + random.random()
    print('gen_w:{0},gen_b:{1}'.format(w,b))
    num_sample = 100
    x = np.random.randint(1, 50, num_sample)
    y = inference(w, b, x) + random.random() * 5
    return x, y

def run():
    x, y = gen_sample_data()
    batch_size = 50
    lr = 0.1
    max_iter = 300
    train(x, y, batch_size, lr, max_iter)

if __name__=='__main__':
    run()
