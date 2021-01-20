import pandas as pd
import numpy as np
import re

import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load graduate and undergraduate data
data_loc = '/Users/mingyupark/spyder/plc_grad/data/'
df_grad = pd.read_excel(data_loc + '대학원생설문결과(중복제거).xlsx').iloc[:,1:]
df_ungrad = pd.read_csv(data_loc + '학부생설문결과(중복제거).csv', encoding='cp949').iloc[:,1:]

# Extract only usefull columns
# Create a column for each independent variable
df_grad['indp_var'] = ['grad'] * len(df_grad)
whether_grad = []
for i in df_ungrad['석사진학계획']:
    if i == '2) 없다':
        whether_grad.append('no_wish')
    else:
        whether_grad.append('wish')
df_ungrad['indp_var'] = whether_grad

anova_cols = sorted(list(set.intersection(set(df_grad.columns), set(df_ungrad.columns))))
useless = ['석사진학계기', '현재학기', '대학원인식_기타', '취업_기타']
anova_cols = [i for i in anova_cols if i not in useless]

# Concat data
df = pd.concat([df_grad[anova_cols], df_ungrad[anova_cols]], axis=0)

# Change values with score
score_like5 = ['매우원하지않음', '원하지않음', '보통', '원함', '매우원함']
score_5 = ['매우그렇지않다', '그렇지않다', '보통이다', '그렇다', '매우그렇다']
not_score_cols = ['indp_var', '본가위치', '학과졸업생교수여부', '전공분류', '자대학원출신교수여부']
like_cols = ['취업_관리직', '취업_기술직', '취업_문화관련직', '취업_사무직', 
'취업_생산직', '취업_연구직', '취업_영업직']
dontknow_cols = ['주변대학원인식', '분야대학원학위요구']
score_cols = list(set(df.columns) - set(not_score_cols) - set(like_cols) - set(dontknow_cols))

for i in df.columns:
    new_values = []
    for e, j in enumerate(df[i]):
        if i == '교수진로상담':
            if e > 144:
                j = j[3:]
                j = re.sub('[^0-9]', '', j)
            else:
                j = re.sub('[^0-9]', '', j)
            j = int(j)
        elif i in dontknow_cols:
            j = re.sub('[^a-zA-Zㄱ-힑]', '', j)
            if j == '잘모른다':
                j = 3
            else:
                j = score_5.index(j) + 1
        elif i in like_cols:
            if type(j) is float:
                j = 3
            else:
                j = re.sub('[^a-zA-Zㄱ-힑]', '', j)
                j = score_like5.index(j) + 1
        elif i in score_cols:
            j = re.sub('[^a-zA-Zㄱ-힑]', '', j)
            j = score_5.index(j) + 1
        elif i in not_score_cols[1:]:
            j = re.sub('[^a-zA-Zㄱ-힑]', '', j)
        new_values.append(j)
    df[i] = new_values


# ANOVA and Chi-square Test
numerical = list(set(df.columns) - set(not_score_cols))
categorical  = not_score_cols[1:]

# ANOVA for numerical y variables
result_df = pd.DataFrame()
for col in numerical:
    model = ols('{} ~ C(indp_var)'.format(col), df).fit()
    anova_df = anova_lm(model)
    name = pd.DataFrame([col] + [' '] * 4).T
    name.columns = anova_df.columns
    result_df = pd.concat([result_df, name, anova_df], axis=0)

# Chi-square for categorical y variables
chi_score = []
p_value = []
degree_f = []
for col in categorical:
    cross_df = pd.crosstab(df[col], df['indp_var'])
    chi_vals = stats.chi2_contingency(observed=cross_df)
    chi_score.append(chi_vals[0])
    p_value.append(chi_vals[1])
    degree_f.append(chi_vals[2])
chi_df = pd.DataFrame({'chi_score' : chi_score, 
                       'p_value' : p_value,
                       'DF' : degree_f}, index=categorical)

result_df.to_csv(data_loc + 'ANOVA_result0120.csv')
chi_df.to_csv(data_loc + 'Chi_result0120.csv')