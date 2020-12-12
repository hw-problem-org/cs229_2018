from p01b_logreg import main as p01b

p01b(train_path='../data/ds1_train.csv',
     eval_path='../data/ds1_valid.csv',
     pred_path='output/p01b_pred_1.txt')

p01b(train_path='../data/ds2_train.csv',
     eval_path='../data/ds2_valid.csv',
     pred_path='output/p01b_pred_2.txt')

from p01e_gda import main as p01e

p01e(train_path='../data/ds1_train.csv',
     eval_path='../data/ds1_valid.csv',
     pred_path='output/p01e_pred_1.txt')

p01e(train_path='../data/ds2_train.csv',
     eval_path='../data/ds2_valid.csv',
     pred_path='output/p01e_pred_2.txt')

from p02cde_posonly import main as p02
p02(train_path='../data/ds3_train.csv',
    valid_path='../data/ds3_valid.csv',
    test_path='../data/ds3_test.csv',
    pred_path='output/p02X_pred.txt')

from p03d_poisson import main as p03
p03(lr=1e-7,
    train_path='../data/ds4_train.csv',
    eval_path='../data/ds4_valid.csv',
    pred_path='output/p03d_pred.txt')
