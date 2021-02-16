import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


# Load data
data_loc = '/Users/mingyupark/spyder/plc_grad/data/'
df_all = pd.read_csv(data_loc + 'survey_under.csv', encoding='cp949').iloc[:,1:]


# Run model
# For C: smaller values specify stronger regularization
def logistic(table):
    X = StandardScaler().fit_transform(np.array(table.drop('석사진학계획', axis=1)))
    X_cols = table.drop('석사진학계획', axis=1).columns
    y = table['석사진학계획']
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)
    lasso = LogisticRegression(penalty='l1', C=1, solver='liblinear')
    lasso.fit(X_tr, y_tr)

    y_pred = lasso.predict(X_ts)

    print('accuracy:', np.round(accuracy_score(y_ts, y_pred), 3))
    print('precision:', np.round(precision_score(y_ts, y_pred), 3))
    print('recall:', np.round(recall_score(y_ts, y_pred), 3))
    print('f1 score:', np.round(f1_score(y_ts, y_pred), 3))

    # Coefficient
    # ce_dict = dict(zip(X_cols, np.round(np.abs(lasso.coef_[0]), 3)))
    ce_dict = dict(zip(X_cols, np.round(lasso.coef_[0], 3)))
    sort_ce = {k: v for k, v in sorted(ce_dict.items(), key=lambda x: np.abs(x[1]), reverse=True)}
    exp_dict = dict(zip(X_cols, np.round(np.exp(np.abs(lasso.coef_[0])), 3)))
    sort_exp = {k: v for k, v in sorted(exp_dict.items(), key=lambda x: x[1], reverse=True)}

    new_cols = ['x_{}'.format(i) for i in range(len(X_cols))]
    print(new_cols)
    table_ = pd.DataFrame(X, columns=new_cols)
    table_['x_36'] = y
    formula = 'x_36 ~ {}'.format(' + '.join(new_cols[:-1]))
    model = sm.GLM.from_formula(formula, family=sm.families.Binomial(), data=table_)
    result = model.fit()
    pval_dict = dict(zip(X_cols, result.pvalues))
    for i in sort_exp.keys():
        print(i, np.round(pval_dict[i], 5))

    return sort_ce, sort_exp


# There are 3 times of analysis with 3 variable sets
# Set 1. Using all variables
set1_ce, set1_exp = logistic(df_all)

# Set 2. Using important features
set2_cols = ['취업시석사학력도움여부', '취업_영업직', '취업_기술직', '취업_연구직', '대학원에좋지않은인식',
 '분야대학원학위요구', '대학원안좋은인식_플젝업무', '교수진로상담', '석사진학계획', 'RA활동여부', '전공만족도',
 '대학원홍보물접한여부']
set2_df = df_all[set2_cols]
set2_ce, set2_exp = logistic(set2_df)

# Set 3. Using all variables except '분야대학원학위요구', '취업_{}'
set3_cols = [i for i in df_all.columns if '취업_' not in i][1:]
set3_df = df_all[set3_cols]
set3_ce, set3_exp = logistic(set3_df)

