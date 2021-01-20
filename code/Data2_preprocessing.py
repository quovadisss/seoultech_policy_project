import pandas as pd
import numpy as np

# Unique data
data_loc = '/Users/mingyupark/spyder/plc_grad/data/'
df = pd.read_excel(data_loc + '대학원생설문결과.xlsx')
print('original length:', len(df))
print('unique lenght:', len(np.unique(df[df.columns[-1]])))

df = df.drop_duplicates(subset=['사례품 발송을 위해 휴대전화 번호를 수집하고 있습니다.(010-XXXX-XXXX의 형식으로 입력해주세요)'])
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


# Save dataframe
df.to_excel(data_loc + '대학원생설문결과(중복제거).xlsx')

