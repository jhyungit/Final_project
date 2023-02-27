# # content_based_v2 by jh

# 필요 패키지 import
import time
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity

start = time.time()

# Data Load
able_food_makers = pd.read_csv('./data/pps_data/able/able_food_makers.csv',index_col = 0)
able_user_group = pd.read_csv('./data/pps_data/able/able_user_group.csv',index_col = 0)
order_food_data = pd.read_csv('./data/raw_data/order/order_raw.csv', index_col=0)
foodtag = pd.read_csv('./data/raw_data/food/foodtag_raw.csv',index_col=0)

# Data Handling
makers_set = set(able_food_makers['MakersId'].values)
food_set = set(able_food_makers['FoodId'].values)
user_set = set(able_user_group['UserId'].values)
group_set = set(able_user_group['GroupId'].values)

g_m = order_food_data[['Created','GroupId', 'MakersId']]
g_m.sort_values(['Created','GroupId'], ascending=False, inplace=True, ignore_index=True)
u_f = order_food_data[['Created','UserId', 'FoodId']]
u_f.sort_values(['Created','FoodId'], ascending=False, inplace=True, ignore_index=True)
u_m = order_food_data[['Created','UserId', 'MakersId']]
u_m.sort_values(['Created','MakersId'], ascending=False, inplace=True, ignore_index=True)

recently_n_group_makers_dic = {}
recently_n_user_foods_dic = {}
recently_n_user_makers_dic = {}

# group별 최근  이용한 N개의 makers가 있는 dict (1<=N<=5) 여기서 N은 5
def make_recent_group_makers(n=5):
    for g in group_set:
        recently_n_group_makers_dic[g] = set({})
    for g in group_set:
        for m in g_m[g_m['GroupId']==g]['MakersId'].values:
            if m in makers_set and len(recently_n_group_makers_dic[g]) != n:
                recently_n_group_makers_dic[g].add(m)
            else:
                pass

# user별 최근 주문한 N개의 foodid가 있는 dict (1<=N<=5) 여기서 N은 5
def make_recent_user_food(n=5):
    for u in user_set:
        recently_n_user_foods_dic[u] = set({})
    for u in user_set:
        for f in u_f[u_f['UserId']==u]['FoodId'].values:
            if f in food_set and len(recently_n_user_foods_dic[u]) != n:
                recently_n_user_foods_dic[u].add(f)
            else:
                pass

# user별 최근 이용한 N개의 makersid가 있는 dict (1<=N<=5) 여기서 N은 5
def make_recent_user_makers(n=5):
    for u in user_set:
        recently_n_user_makers_dic[u] = set({})
    for u in user_set:
        for m in u_m[u_m['UserId']==u]['MakersId'].values:
            if m in makers_set and len(recently_n_user_makers_dic[u]) != n:
                recently_n_user_makers_dic[u].add(m)
            else:
                pass
 
make_recent_group_makers()
make_recent_user_food()
make_recent_user_makers()

foodid_makersid = able_food_makers[['FoodId','MakersId']]
makerstag = pd.merge(foodid_makersid,foodtag, how = 'inner')
makerstag.drop('FoodId',inplace=True, axis = 1)

# makersid-makersid CS
def make_mm_cs():
    mk_df = pd.DataFrame({})

    for mkid in tqdm(makerstag['MakersId']):
        temp_df = pd.DataFrame({})
        temp_df = makerstag[makerstag['MakersId'] == mkid].groupby('MakersId').sum()
        mk_df = pd.concat([mk_df, temp_df], axis = 0)

    mk_df = mk_df.loc[~mk_df.index.duplicated(keep='first')]

    norm = Normalizer()
    mk_df.iloc[:,:] = norm.fit_transform(mk_df.iloc[:,:])
    makers_cs_df = cosine_similarity(mk_df, mk_df)
    makers_cs_df = pd.DataFrame(makers_cs_df, index = mk_df.index, columns= mk_df.index)
    return makers_cs_df

# foodid-foodid CS
def make_ff_cs():
    foodid_makersid = pd.DataFrame()

    for f in food_set:
        foodid_makersid = pd.concat([foodid_makersid,foodtag[foodtag['FoodId'] == f]], ignore_index=True)

    foodid_makersid.set_index('FoodId', inplace=True)

    food_cs_df = cosine_similarity(foodid_makersid, foodid_makersid)
    food_cs_df = pd.DataFrame(food_cs_df, index = foodid_makersid.index, columns= foodid_makersid.index)
    return food_cs_df
    
# GroupId, MakersId, Score DF 만들기 : 행은 Makersid, 열은 GroupId
group_makers_recom_score = pd.DataFrame(index=[g for g in group_set], columns=[m for m in makers_set])
group_makers_recom_score.fillna(0, inplace=True)
# UserId, FoodId, Score DF 만들기 : 행은 Userid, 열은 Foodid
user_food_recom_score = pd.DataFrame(index=[u for u in user_set], columns=[f for f in food_set])
user_food_recom_score.fillna(0, inplace=True)
# UserId, MakersId, Score DF 만들기 : 행은 Makersid, 열은 GroupId
user_makers_recom_score = pd.DataFrame(index=[g for g in user_set], columns=[m for m in makers_set])
user_makers_recom_score.fillna(0, inplace=True)

# 그룹이 최근 이용한 N개의 메이커스 기반 메이커스 추천
def by_n_group_recently_used_makers(makersid):
    global group_makers_recom_score
    # 주어진 메이커스와 다른 메이커스의 similarity를 가져온다
    makers_cs_df = make_mm_cs()
    sim_scores = makers_cs_df.loc[[m for m in makersid]].apply(lambda x: sum(x), axis = 0)
    sim_scores = sim_scores.apply(lambda x : x/len(makersid))
    
    for m_score, m_col in zip(sim_scores.values, sim_scores.index):
        group_makers_recom_score.loc[g, m_col] = m_score

# 사용자가 가장 최근에 먹은 음식 기반 음식 추천
def by_n_user_recently_ordered_food(foodid):
    global user_food_recom_score
    # 주어진 음식과 다른 음식의 similarity를 가져온다
    food_cs_df = make_ff_cs()
    sim_scores = food_cs_df.loc[[f for f in foodid]].apply(lambda x: sum(x), axis = 0)
    sim_scores = sim_scores.apply(lambda x : x/len(foodid))
    
    for f_score, f_col in  zip(sim_scores.values, sim_scores.index):
        user_food_recom_score.loc[u, f_col] = f_score

# 유저가 최근 이용한 N개의 메이커스 기반 메이커스 추천
def by_n_user_recently_used_makers(makersid):
    global user_makers_recom_score
    # 주어진 메이커스와 다른 메이커스의 similarity를 가져온다
    makers_cs_df = make_mm_cs()
    sim_scores = makers_cs_df.loc[[m for m in makersid]].apply(lambda x: sum(x), axis = 0)
    sim_scores = sim_scores.apply(lambda x : x/len(makersid))
    
    for m_score, m_col in zip(sim_scores.values, sim_scores.index):
        user_makers_recom_score.loc[u, m_col] = m_score
        
# 위에서 만든 group_makers_recom_df, ser_food_recom_df, group_makers_recom_df에 value값 저장
for g in tqdm(group_set):
    if recently_n_group_makers_dic[g] != set():
        by_n_group_recently_used_makers(recently_n_group_makers_dic[g])
for u in tqdm(user_set):
    if recently_n_user_foods_dic[u] != set():
        by_n_user_recently_ordered_food(recently_n_user_foods_dic[u])
        by_n_user_recently_used_makers(recently_n_user_makers_dic[u])

all_group_makers_recom = pd.DataFrame(columns = ['GroupId','MakersId','Score'])
all_user_food_recom = pd.DataFrame(columns = ['UserId', 'FoodId', 'Score'])
all_user_makers_recom = pd.DataFrame(columns = ['UserId', 'MakersId', 'Score'])

#행은 GroupId, 열은 Makersid
for i in (sorted(group_makers_recom_score.index)):
    temp = pd.DataFrame(columns = ['GroupId','MakersId','Score'])
    temp['GroupId'] = [i] * len(group_makers_recom_score.columns)
    temp['MakersId'] = group_makers_recom_score.loc[i].sort_values(ascending=False).index
    temp['Score'] = group_makers_recom_score.loc[i].sort_values(ascending=False).values
    all_group_makers_recom = pd.concat([all_group_makers_recom, temp], ignore_index=True)

#행은 Userid, 열은 Foodid
for i in (sorted(user_food_recom_score.index)):
    temp = pd.DataFrame(columns = ['UserId', 'FoodId', 'Score'])
    temp['UserId'] = [i] * len(user_food_recom_score.columns)
    temp['FoodId'] = user_food_recom_score.loc[i].sort_values(ascending=False).index
    temp['Score'] = user_food_recom_score.loc[i].sort_values(ascending=False).values
    all_user_food_recom = pd.concat([all_user_food_recom, temp], ignore_index=True)

#행은 UserId, 열은 Makersid
for i in (sorted(user_makers_recom_score.index)):
    temp = pd.DataFrame(columns = ['UserId', 'MakersId', 'Score'])
    temp['UserId'] = [i] * len(user_makers_recom_score.columns)
    temp['MakersId'] = user_makers_recom_score.loc[i].sort_values(ascending=False).index
    temp['Score'] = user_makers_recom_score.loc[i].sort_values(ascending=False).values
    all_user_makers_recom = pd.concat([all_user_makers_recom, temp], ignore_index=True)

all_group_makers_recom.to_csv('./model/contents_based_v2/results/group_makers_score.csv')
all_user_food_recom.to_csv('./model/contents_based_v2/results/user_food_score.csv')
all_user_makers_recom.to_csv('./model/contents_based_v2/results/user_makers_score.csv')

end = time.time()
print('실행시간:',end-start)