import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


# Load data
data_loc = '/Users/mingyupark/spyder/plc_grad/data/'
df_all = pd.read_csv(data_loc + 'survey_under.csv', encoding='cp949').iloc[:,1:]
cols = ['대학원에좋지않은인식', '취업시석사학력도움여부', '대학원안좋은인식_플젝업무', '취업_영업직',
  '취업_생산직', '취업_관리직', '취업_문화관련직', '현재학기', '대학내금전적어려움', '대학원안좋은인식_등록금',
  '교수진로상담', '취업_사무직', '본인학부공부성실도', '전공만족도', '교수수업만족', '커리큘럼만족', 
  '대학원안좋은인식_연구', '대학원안좋은인식_개인시간', '석사진학계획']
df = df_all[cols]

# Check null value and dtypes
print(df.isnull().sum())
print(df.dtypes)

# Split data with normalization
X = StandardScaler().fit_transform(np.array(df.iloc[:,:-1]))
y = df.iloc[:,-1]
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=0)

# Run model
# For C: smaller values specify stronger regularization
lasso = LogisticRegression(penalty='l1', C=1, solver='liblinear')

lasso.fit(X_tr, y_tr)
imp = np.round(np.abs(lasso.coef_), 3)[0].tolist()
imp_dict = dict(zip(imp, list(df.columns)))
sort_dict = {k: v for k, v in sorted(imp_dict.items(), 
                                     key=lambda x: x[0], 
                                     reverse=True)}

y_pred = lasso.predict(X_ts)

print('accuracy:', np.round(accuracy_score(y_ts, y_pred), 3))
print('precision:', np.round(precision_score(y_ts, y_pred), 3))
print('recall:', np.round(recall_score(y_ts, y_pred), 3))
print('f1 score:', np.round(f1_score(y_ts, y_pred), 3))