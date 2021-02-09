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


# 대학원생 데이터 전처리
def graduate_survey_pre(df):
    df = df.drop_duplicates(subset=['사례품 발송을 위해 휴대전화 번호를 수집하고 ' + \
        '있습니다.(010-XXXX-XXXX의 형식으로 입력해주세요)'])
    df = df[df[df.columns[-1]] != '010']

    # Make number 
    new_num = []
    for i in df[df.columns[-1]]:
        if len(i) > 4:
            if '-' in i:
                new_num.append(i)
            else:
                new_num.append(i[:3] + '-' + i[3:7] + '-' + i[7:])
        else:
            new_num.append(i)

    df = df.drop([df.columns[-1], df.columns[0]], axis=1)

    # Rename columns
    new_grad_cols = ['본가위치', '전공분류', '취업_연구직', '취업_기술직', '취업_사무직', '취업_생산직', 
    '취업_영업직', '취업_관리직', '취업_문화관련직', '취업_기타', '분야대학원학위요구', '대학원인식', 
    '주변대학원인식', '대학원인식_등록금', '대학원인식_연구', '대학원인식_플젝업무', '대학원인식_개인시간', 
    '대학원인식_교수비인격적', '대학원인식_교수사적업무', '대학원인식_진로불투명', '대학원인식_기타', '교수수업만족',
    '교수진로상담', '장학금만족', '위치만족', '시설만족', '커리큘럼만족', '자대학원출신교수여부', '학과졸업생교수여부', 
    '자대일치여부', '자대진학이유', '과기대진학결정이유', '현재학기', '석사진학계기', '대학원진학결정시기', 
    '연관학과여부', '석사후취업긍정도', '최종학위목표', '원졸업후취업_사기업', '원졸업후취업_공기업', 
    '원졸업후취업_연구소', '원졸업후취업_교육기관', '원졸업후취업_국가기관', '원졸업후취업_기타']

    df.columns = new_grad_cols
    df['Phone'] = new_num

    return df