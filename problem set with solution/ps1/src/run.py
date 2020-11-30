from p01b_logreg import main as p01b

p01b(train_path='../data/ds1_train.csv',
     eval_path='../data/ds1_valid.csv',
     pred_path='output/p01b_pred_1.txt')

p01b(train_path='../data/ds2_train.csv',
     eval_path='../data/ds2_valid.csv',
     pred_path='output/p01b_pred_2.txt')
