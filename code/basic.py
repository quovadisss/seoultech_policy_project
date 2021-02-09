import pandas as pd
from utils import *


# Data location
data_loc = '/Users/mingyupark/spyder/plc_grad/data/'

"""
Data 1. 학적데이터
데이터 설명 : 자대 대학원생의 학부 학적 데이터
데이터 전처리 업무 : 성적, 휴학 column 정리
"""
# 학기별 성적 및 성적 파생변수 생성
df_1 = pd.read_excel(data_loc + '학기별성적_일반대학원_본교(2011~2014).xls')
df_2 = pd.read_excel(data_loc + '학기별성적_일반대학원_본교(2015~2020).xls')

score_df = scores_pre(df_1, df_2)
derived_df = scores_derived(score_df)
derived_df.to_csv(data_loc + '학기별성적(파생포함).csv', encoding='cp949')


# 휴학 column 전처리
df_3 = pd.read_excel(data_loc + '휴학현황_일반대학원_본교.xls')
absence_df = absence_sem(df_3)
absence_df.to_csv(data_loc + '휴학전처리.csv', encoding='cp949')



"""
Data 2. 설문데이터
데이터 설명 : 자대 학부생 설문 데이터(학부/대학원)
데이터 전처리 업무 : 구글 설문 데이터 -> 대학원생 데이터 전처리 
"""
df_4 = pd.read_excel(data_loc + '대학원생설문결과.xlsx')
sur_df = graduate_survey_pre(df_4)
sur_df.to_excel(data_loc + '대학원생설문결과(중복제거).xlsx')
