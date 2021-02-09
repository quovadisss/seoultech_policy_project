import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LinearRegression


# 학기별 성적 전처리
def scores_pre(df_1, df_2):
    # Concat data
    df = pd.concat([df_1, df_2], axis=0)

    # extract only scores, semester, and id
    arr = np.array(df[['개설학년도학기', '평점평균', '학생통계번호']])
    year_sem = sorted(np.unique(df['개설학년도학기']))

    uniq_id_b = sorted(np.unique(df['학생통계번호']))
    new_df = pd.DataFrame(columns=year_sem, index=uniq_id_b)
    for i in range(len(arr)):
        y = int(arr[i][0])
        s = int(arr[i][2])
        new_df[y][s] = arr[i][1]

    new_col = []
    for i in year_sem:
        new_col.append('학기별평점_{}'.format(i))
    new_df.columns = new_col

    return new_df


# 성적 파생변수    
def scores_derived(df):
    score = []
    mean_score = []
    two = []
    trend = []
    for i in range(len(df)):
        un_s = []
        trend_score = 0
        for j in df.iloc[i, :]:
            if math.isnan(j):
                pass
            elif j != 0:
                un_s.append(j)
        score.append(un_s)
        mean_score.append(np.mean(un_s))

        # lastest two semester
        # Calculate trend score
        if len(un_s) > 1:
            two.append(np.mean(un_s[-2:]))

            obj = -1
            for ind in range(len(un_s)-1):
                trend_score += (un_s[obj] - un_s[obj-1])
                obj -= 1

            trend.append(trend_score)
        else:
            two.append(un_s[-1])
            trend.append(trend_score)

    scores_nonan = []
    for i in score:
        scores_nonan.append([j for j in i if not math.isnan(j)])

    # Calculate linear coefficient
    coeffi = []
    for y in scores_nonan:
        x = np.arange(len(y)).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        coeffi.append(np.round(reg.coef_[0][0], 3))

    # Coefficient for latest 4 semester
    coeffi_4 = []
    for y in scores_nonan:
        if len(y) > 3:
            x = np.arange(4).reshape(-1, 1)
            y = np.array(y[-4:]).reshape(-1, 1)
        else:
            x = np.arange(len(y)).reshape(-1, 1)
            y = np.array(y).reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        coeffi_4.append(np.round(reg.coef_[0][0], 3))

    # Latest 2 semester
    sem_2 = []
    for e in scores_nonan:
        if len(e) == 1:
            sem_2.append(0)
        else:
            sem_2.append(np.round(e[-2:][0] / e[-2:][1] - 1, 3))

    # new score dataframe
    new_ind = [i for i in range(len(df))]
    new_cols = ['{}학기성적'.format(i+1) for i in range(12)]
    new_df = pd.DataFrame(score, columns=new_cols, index=new_ind)
    new_df['학부평균성적'] = mean_score
    new_df['직전2학기성적'] = two
    new_df['성적오름추세'] = trend
    new_df['성적기울기'] = coeffi
    new_df['성적기울기_직전4학기'] = coeffi_4
    new_df['직전2학기증가율'] = sem_2

    return new_df


# 휴학 전처리
def absence_sem(df):
    rest = np.unique(df['휴학사유'])
    uniq_id_b = sorted(np.unique(df['학생통계번호']))

    arr = np.array(df[['휴학사유', '학생통계번호']])

    # Make new dataframe
    new_df = pd.DataFrame(0, columns=rest, index=uniq_id_b)
    for i in range(len(arr)):
        r = arr[i][0]
        s = int(arr[i][1])
        new_df[r][s] += 1

    new_col = []
    for i in rest:
        new_col.append('휴학사유_{}'.format(i))
    new_df.columns = new_col

    # 휴학 yes = 1, 군휴학 = 2, no = 0
    new_df_2 = pd.DataFrame(0, columns=['휴학_기타', '휴학_군대'], index=new_df.index)
    army = [2, 3, 7]
    arr_2 = np.array(new_df)
    for i in range(len(arr_2)):
        for j in range(len(arr_2[0])):
            if j in army:
                if arr_2[i][j] != 0:
                    new_df_2['휴학_군대'][new_df.index[i]] = 1
            else:
                if arr_2[i][j] != 0:
                    new_df_2['휴학_기타'][new_df.index[i]] = 1

    return new_df_2

# # Check whether columns are same
# print(set(df_1.columns) - set(df_2.columns))

# # Concat dataframe
# df = pd.concat([df_1, df_2], axis=0)
# print(df.isnull().sum())

# # Unique students
# uniq_id_b = sorted(np.unique(df['학생통계번호']))
# print(len(uniq_id_b))
# uniq_id_g = np.unique(df['대학원통계번호'])
# print(len(uniq_id_g)) # 학생통계번호와 대학원통계번호 unique값이 다름.




# # Integrated scores
# new_df = pd.read_excel('학기별성적_전처리.xlsx')


# new_df_2.to_excel('학부성적_전처리_정리.xlsx')