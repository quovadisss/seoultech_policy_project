"""
This python code is for clustering on data1.
KMeans and KMedoids are used.
Elbow and Silhouette techniques are used.
TSNE are used for visulization.
"""


# Modlues
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
import pandas_profiling


# 1. Load Data
df = pd.read_excel('~/spyder/plc_grad/data/통합전처리_210105.xls').iloc[:,1:]


# 2. Extract columns will be used
# Useful columns
df = df[['성별', '학부_편입여부', '현장실습이수여부', '복수전공여부',
         '부전공여부', '교환학생여부', '수도권_거주여부', '장학금액', '학습역량 여부',
         '진로,심리 여부', '취업,진로 여부', '창업 여부', '비교과 여부', '휴학_기타',
        '휴학_군대', '학부평균성적', '직전2학기평균', '성적오름추세', '직전2학기증가율',
        '학부입학 후 대학원입학까지 걸린 시간', '학부와 대학원 전공의 계열 일치 여부']]

df['성별'] = df['성별'].astype('category')
df = pd.get_dummies(df)
df.drop(['성별_남자'], inplace=True, axis=1)

# Exam score columns only
df = df[['학부평균성적', '직전2학기평균', '성적오름추세', '직전2학기증가율']]


# 3. Nomalization
# data = np.array(df)
df_nor = MinMaxScaler().fit_transform(np.array(df))


# 4. Plot setting
# Find korean font
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
kor_list = [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Gothic' in f.name]

# Apply the path
path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
fontprop = fm.FontProperties(fname=path, size=13)

# Plot size
plt.rcParams['figure.figsize'] = (14, 10)


# 5. Clustering
# Elbow
def elbow(X, model, name):
    sse = []
    for i in range(1, 11):
        km = model(n_clusters=i, random_state=0)
        km.fit(X)
        sse.append(km.inertia_)
        
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('클러스터 개수', fontproperties=fontprop)
    plt.ylabel('SSE')
    plt.savefig('/Users/mingyupark/spyder/seoultech_policy_project'+
        '/data/output/elbow_{}.png'.format(name))
    plt.show()


# Silhouette
def silhouette(n_clusters, data, model, name):
    km = model(n_clusters=n_clusters, random_state=0)
    y_km = km.fit_predict(data)
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(data, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i/n_clusters)
        
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)
        
    silhoutte_avg = np.mean(silhouette_vals)
    plt.axvline(silhoutte_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels+1)
    plt.ylabel('클러스터', fontproperties=fontprop)
    plt.xlabel('실루엣 계수', fontproperties=fontprop)
    plt.savefig('/Users/mingyupark/spyder/seoultech_policy_project'+
        '/data/output/slhtt_{0}_{1}.png'.format(n_clusters, name))
    plt.show()
    
    return km


# Split dataframe by each label
def split_df(n_clusters, data, model, name):
    km = silhouette(n_clusters, data, model, name)
    col_name = 'km_{}'.format(n_clusters)
    df[col_name] = km.labels_
    
    df_all = []
    for i in range(n_clusters):
        df_ = df[df[col_name]==i]
        df_all.append(df_)
            
    return df_all, km


# KMeans
elbow(df_nor, KMeans, 'KMeans')
df_all, km = split_df(5, df_nor, KMeans, 'KMeans')

# KMedoids
# elbow(df_nor, KMeans, 'KMedoids')
# df_all, km = split_df(5, df_nor, KMedoids, 'KMedoids')

# Save df with label
df.to_excel('/Users/mingyupark/spyder/plc_grad'+
        '/data/output/통합kmeans.xlsx') # or kmedoids

# 6. Make comparison dataframe
numerical = ['장학금액', '학부평균성적', '직전2학기평균', '성적오름추세', '직전2학기증가율',
             '학부입학 후 대학원입학까지 걸린 시간']
useless = ['km_{}'.format(i) for i in range(3, 8)]
binary = sorted(list(set(df.columns) - set(numerical) - set(useless)))


def comp(bina, nume, dfs, name):
    final_df = pd.DataFrame()

    for b in bina:
        print(b)
        l = [''] * len(dfs) * 2
        l[0] = b
        df_v_all = pd.DataFrame(l).T
        col_count = ['count_{}'.format(i) for i in range(len(dfs))]
        col_ratio = ['ratio_{}'.format(i) for i in range(len(dfs))]
        cols = col_count + col_ratio
        df_v_all.columns = cols
        
        df_v_all_2 = pd.DataFrame()
        for n, df_ in enumerate(dfs):
            count = df_[b].value_counts()
            ratio = []
            for i in range(len(count)):
                try:
                    ratio.append(count[i]/count.sum())
                except KeyError:
                    ratio.append(count[i+1]/count.sum())
            df_v = pd.DataFrame({'count_{}'.format(n):count, 'ratio_{}'.format(n):ratio})
            df_v_all_2 = pd.concat([df_v_all_2, df_v], axis=1)

        df_v_all = pd.concat([df_v_all, df_v_all_2], axis=0)

        final_df = pd.concat([final_df, df_v_all], axis=0)

    final_df.to_csv('/Users/mingyupark/spyder/plc_grad'+
            '/data/output/count_{0}_{1}.csv'.format(len(dfs) ,name), encoding='euc-kr')
    # numerical = ['학부평균성적', '직전2학기평균', '성적오름추세', '직전2학기증가율']

    final_df_n = pd.DataFrame()
    for nu in nume:
        l = [''] * len(dfs)
        l[0] = nu
        df_v_all = pd.DataFrame(l).T
        df_v_all_2 = pd.DataFrame()
        for n, df_ in enumerate(dfs):
            df_v_all_2[n] = df_[nu].describe()
        df_v_all = pd.concat([df_v_all, df_v_all_2])
        
        final_df_n = pd.concat([final_df_n, df_v_all])

    final_df_n.to_csv('/Users/mingyupark/spyder/plc_grad'+
            '/data/output/numerical_{0}_{1}.csv'.format(len(dfs), name), encoding='euc-kr')


comp(binary, numerical, df_all, 'KMeans')
# comp(binary, numerical, df_all, 'KMedoids')



# 7. Visualization using TSNE
def visual_2d(data, n_clusters, name):
    tsne = TSNE(learning_rate=100)
    transformed = tsne.fit_transform(data)

    plt.scatter(transformed[:,0], transformed[:,1], c=df['km_5'])
    plt.legend()
    plt.savefig('/Users/mingyupark/spyder/seoultech_policy_project'+
            '/data/output/TSNE_{0}_{1}'.format(n_clusters, name))


visual_2d(df_nor, len(df_all), 'KMeans')
# visual_2d(df_nor, len(df_all), 'KMedoids')


# Pandas profiling
new_df = pd.read_excel('/Users/mingyupark/spyder/plc_grad'+
        '/data/output/통합kmeans_all.xlsx')

print(new_df.dtypes)

need_change = ['학부_편입여부', '학부_입학년도', '현장실습이수여부', '복수전공여부',
       '부전공여부', '교환학생여부', '수도권_거주여부', '학습역량 여부', '진로,심리 여부',
       '취업,진로 여부', '창업 여부', '비교과 여부', '휴학_기타', '휴학_군대', '대학원_입학연도']
for i in need_change:
    new_df[i] = new_df[i].astype('category')

for j in range(5):
    pr = new_df[new_df['km_5'] == j].profile_report()
    pr.to_file('/Users/mingyupark/spyder/plc_grad'+
        '/data/output/km_profile_{}.html'.format(j))