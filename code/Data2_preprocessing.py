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

df = df.drop([df.columns[-1]], axis=1)
df['Phone'] = new_num

# Save dataframe
df.to_excel(data_loc + '대학원생설문결과(중복제거).xlsx')

