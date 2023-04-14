#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder
from warnings import filterwarnings
from datetime import datetime
filterwarnings("ignore")


# In[2]:


#python kütüphane yükleme

import numpy as np 
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,recall_score,f1_score
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import scale,StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")


# In[172]:


# verileri yükleyelim;


# In[39]:


df_train=pd.read_csv("gbbva_train_csv.txt")
df_train.head()


# In[40]:


df_test=pd.read_csv("gbbva_test_csv.txt")
df_test.head(10)


# In[41]:


df_work=pd.read_csv("gbbva_work_csv.txt")
df_work.head(10)


# In[42]:


df_exp = df_work[df_work['start_year_month'] < 201901]
df_exp


# In[43]:


df_edu=pd.read_csv("gbbva_edu_csv.txt")
df_edu.head(10)


# In[44]:


df_lang=pd.read_csv("gbbva_language_csv.txt")
df_lang.head(10)


# In[45]:


df_skil=pd.read_csv("gbbva_skills_csv.txt")
df_skil.head(10)


# In[173]:


# eda ve veri önişleme


# In[47]:


df_work = df_work.loc[df_work["start_year_month"].lt(201901)]


# In[1]:


# eksik değerleri giderelim;


# In[48]:


df_edu = df_edu.loc[df_edu["degree"].notnull() & df_edu["fields_of_study"].notnull()]


# In[2]:


# kolonları düzeltelim;


# In[49]:


fixed_degree = {
    "(?i).*(High School|Highschool|Lise|Lise Mezunu|Lisemezunu).*":"HighSchool",
    "(?i).*(Ön|Önlisan|Ön lisanas|Associate's degree|Associate).*":"Associate",
    "(?i).*(Lisans|bachelors|bachelor|BS|BS.c|Bsc|B.s|mezun|graduate|engineer|BA|Licence).*":"Bachelors",
    "(?i).*(yüksek|yüksek lisans|master|masters|Msc|MS|MS.c|Masters's degree|M.S).*" : "Masters",
    "(?i).*(Doktora|doctorate|phd|Ph.d).*" :"Doctorate"
}

fixed_university = {
    "(?i).*(Middle East Technical University|Orta Doğu Teknik Üniversitesi|ODTU|METU).*":"Tier1",
    "(?i).*(Hacettepe University|Hacettepe Üniversitesi).*":"Tier1",
    "(?i).*(Boğaziçi|Bogazici|Bogaziçi).*":"Tier1",
    "(?i).*(Ankara University|Ankara Üniversitesi).*":"Tier1",
    "(?i).*(Koç University|Koç Üniversitesi).*":"Tier1",
    "(?i).*(Bilkent University|Bilkent Üniversitesi).*":"Tier1",
    "(?i).*(Istanbul University|Istanbul Üniversitesi).*":"Tier1",
    "(?i).*(Atatürk Üniversitesi|Atatürk University).*":"Tier1",
    "(?i).*(Gazi University|Gazi Üniversitesi).*":"Tier1",
    "(?i).*(Ege University|Ege Üniversitesi).*":"Tier1",
    "(?i).*(Dokuz Eylül University|Dokuz Eylül Üniversitesi).*":"Tier1",
    "(?i).*(Sabanci University|Sabanci Üniveritesi).*":"Tier1",
    "(?i).*(Yildiz Technical University|Yıldız Teknik Üniversitesi|YTÜ).*":"Tier1",
    "(?i).*(University of Anatolia|Anadolu Üniversitesi).*":"Tier1",
    "(?i).*(Marmara University|Marmara Üniversitesi).*":"Tier1",
    "(?i).*(Firat University|Fırat Üniversitesi).*":"Tier1",
    "(?i).*(Istanbul Bilgi University|İstanbul Bilgi Üniversitesi).*":"Tier1",
    "(?i).*(Erciyes University|Erciyes Üniversitesi).*":"Tier1",
    "(?i).*(Sakarya University|Sakarya Üniversitesi).*":"Tier1",
    "(?i).*(Akdeniz University|Akdeniz Üniversitesi).*":"Tier1",
    "(?i).*(Ondokuz Mayis University|Ondokuz Mayıs Üniversitesi).*":"Tier1",
    "(?i).*(Uludağ|Uludag).*":"Tier1",
    "(?i).*(Karabük Üniversitesi:Karabük University).*":"Tier1",
    "(?i).*(Karadeniz Technical University|Karadeniz Teknik Üniversitesi).*":"Tier1",
    "(?i).*(Fırat University|Fırat Üniversitesi).*":"Tier1",
    "(?i).*(Istanbul Technical|İstanbul Teknik Üniversitesi|İstanbul Teknik).*":"Tier1",   
    "(?i).*(Bahçeşehir Üniversitesi|Bahcesehir University|BAU).*":"Tier2",
    "(?i).*(Çankaya University|Çankaya Üniversitesi|Cankaya University).*":"Tier2",
    "(?i).*(Gaziantep University|Gaziantep Üniversites).*":"Tier2",
    "(?i).*(Izmir Institute of Technology|İzmir Yüksek Teknoloji Enstitüsü).*":"Tier2",
    "(?i).*(Eskişehir Osmangazi University|Eskişehir Osmangazi  Üniversitesi).*":"Tier2",
    "(?i).*(Özyeğin University|Ozyegin University|Özyeğin Üniversitesi).*":"Tier2",
    "(?i).*(Cumhuriyet University|Sivas Cumhuriyet Üniversitesi|Cumhuriyet Üniversitesi).*":"Tier2",
    "(?i).*(Gebze Technological University|Gebze Teknik Üniversitesi).*":"Tier2",
    "(?i).*(Suleyman Demirel University|Süleyman Demirel Üniversitesi|Süleyman Demirel University|University of Suleyman Demirel).*":"Tier2",
    "(?i).*(Istanbul Aydin University|İstanbul Aydın Üniversitesi|Istanbul Aydın University).*":"Tier2",
    "(?i).*(TOBB Ekonomi ve Teknoloji Üniversitesi|TOBB University of Economics and Technology).*":"Tier2",
    "(?i).*(Mersin University|Mersin Üniversitesi).*":"Tier2",
    "(?i).*(Pamukkale University|Pamukkale Üniversitesi).*":"Tier2",
    "(?i).*(Kocaeli University|Kocaeli Üniversitesi).*":"Tier2",
    "(?i).*(Harran University|Harran Üniversitesi).*":"Tier2",
    "(?i).*(Yüzüncü Yil University|Yüzüncü Yil Üniversitesi|Yüzüncü Yıl Üniversitesi).*":"Tier2",
    "(?i).*(Trakya University|Trakya Üniversitesi).*":"Tier2",
    "(?i).*(Çanakkale Onsekiz Mart University|Çanakkale Onsekiz Mart Üniversitesi|Canakkale Onsekiz Mart University).*":"Tier2",
    "(?i).*(İnönü University Malatya|İnönü Üniversitesi|İnönü University).*":"Tier2",
    "(?i).*(Başkent University|Başkent Üniversitesi).*":"Tier2",
    "(?i).*(Düzce University|Düzce Üniversitesi).*":"Tier2",
    "(?i).*(Abant Izzet Baysal University|Abant Izzet Baysal Üniversitesi|Abant İzzet Baysal Üniversitesi).*":"Tier2",
    "(?i).*(Muğla Sıtkı Koçman University|Muğla Sıtkı Koçman Üniversitesi|Mugla Sitki Kocman University|Mugla University).*":"Tier2",
    "(?i).*(Gaziosmanpaşa University|Gaziosmanpaşa Üniversitesi|Gaziosmanpasa University).*":"Tier2",
    "(?i).*(Istanbul Medipol University|Istanbul Medipol Üniversitesi|İstanbul Medipol Üniversitesi).*":"Tier3",
    "(?i).*(Yeditepe University|Yeditepe Üniversitesi).*":"Tier3",
    "(?i).*(Gaziantep University|Gaziantep Üniversites).*":"Tier3",
    "(?i).*(Necmettin Erbakan Üniversitesi|Necmettin Erbakan University).*":"Tier3",
    "(?i).*(Çukurova University|Çukurova Üniversitesi|Cukurova University).*":"Tier3",
    "(?i).*(Kafkas University|Kafkas Üniversitesi).*":"Tier3",
    "(?i).*(Adnan Menderes University|Adnan Menderes Üniversitesi).*":"Tier3",
    "(?i).*(Atilim University|Atilim Üniversitesi|Atılım Üniversitesi|Atılım University).*":"Tier3",
    "(?i).*(Healh Sciences University Istanbul|Sağlık Bilimleri Üniversitesi).*":"Tier3",
    "(?i).*(Istanbul Aydin University|İstanbul Aydın Üniversitesi|Istanbul Aydın University).*":"Tier3",
    "(?i).*(İstanbul Medeniyet Üniversitesi|İstanbul Medeniyet University|Istanbul Medeniyet Üniversitesi|Istanbul Medeniyet University).*":"Tier3",
    "(?i).*(Afyon Kocatepe University|Afyon Kocatepe Üniversitesi).*":"Tier3",
    "(?i).*(Balikesir University|Balikesir Üniversitesi|Balıkesir University|Balıkesir Üniversitesi).*":"Tier3",     
    "(?i).*(Giresun University|Giresun Üniversitesi).*":"Tier3",
    "(?i).*(Zonguldak Bülent Ecevit University|Zonguldak Bülent Ecevit Üniversitesi|Zonguldak Bulent Ecevıt).*":"Tier3",
    "(?i).*(Selçuk University|Selçuk Üniversitesi|Selcuk Üniversitesi|Selcuk University).*":"Tier3",
    "(?i).*(Yaşar University|Yaşar Üniversitesi|Yasar University|Yasar Üniversitesi).*":"Tier3",
    "(?i).*(Bingöl University|Bingöl Üniversitesi).*":"Tier3",
    "(?i).*(Bozok University|Bozok Üniversitesi).*":"Tier3",
    "(?i).*(Gelişim University|Gelişim Üniversitesi|Gelisim University|Gelisim Üniversitesi).*":"Tier3",
    "(?i).*(Karamanoğlu Mehmetbey University|Karamanoglu Mehmetbey University|Karamanoğlu Mehmetbey Üniversitesi|Karamanoglu Mehmetbey Üniversitesi).*":"Tier3",
    "(?i).*(Kahramanmaraş Sütçü İmam University|Kahramanmaraş Sütçü İmam Üniversitesi|Kahramanmaras Sutcu İmam Üniversitesi).*":"Tier3",
    "(?i).*(Bartin University|Bartin Üniversitesi).*":"Tier3",
    "(?i).*(Izmir University of Economics|İzmir Ekonomi Üniversitesi).*":"Tier3",    
    "(?i).*(Kadir Has University|Kadir Has Üniversitesi).*":"Tier3",
    "(?i).*(Oxford|Kaliforniya|Harvard|Stanford|Cambridge|Massachusetts|Princeton|Kaliforniya| Berkeley|Yale|Chicago|Kolombiya|Imperial|Johns Hopkins|Pensilvanya|ETH Zürih|Pekin|Tsinghua|Toronto|Londra).*":"Tier_Unique" 
}


# In[50]:


df_edu.loc[:,"degree"] = df_edu.loc[:,"degree"].replace(fixed_degree,regex=True)
df_edu.loc[:,"school_name"] = df_edu.loc[:,"school_name"].replace(fixed_university,regex=True)


# In[51]:


df_edu.loc[~df_edu["degree"].isin(fixed_degree.values()),"degree"] = "Other"


# In[52]:


df_edu = df_edu[df_edu["school_name"].isin(fixed_university.values())]


# In[53]:


df_edu.drop_duplicates(["degree","user_id"],inplace=True)


# In[17]:


plt.figure(figsize=(9,5))
sns.barplot(x=df_edu["school_name"].value_counts()[:3].index,y=df_edu["school_name"].value_counts()[:3]);
plt.xlabel("Universities",fontsize=15)
plt.ylabel("Number of People",fontsize=15)
plt.title("The Distribution of Top Three Universities")
plt.show()


# In[19]:


sns.barplot(x=df_edu["degree"].value_counts().index,y=df_edu["degree"].value_counts());
plt.xlabel("Degree",fontsize=15)
plt.ylabel("Number of People",fontsize=15)
plt.title("The Distribution University Degrees")
plt.show()


# In[54]:


df_edu = df_edu.drop(columns=["fields_of_study","start_year_month","end_year_month"])


# In[55]:


df_edu = pd.get_dummies(df_edu,drop_first=True)


# In[56]:


df_edu = df_edu.groupby(by="user_id").sum()


# In[57]:


df_edu = df_edu.astype("object")


# In[58]:


df_edu


# In[24]:


df_edu = df_edu.astype("object")


# In[25]:


df_edu


# In[26]:


# skil data;


# In[59]:


values = list(df_skil["skill"].value_counts()[:10].index)


# In[60]:


temp_df = pd.DataFrame({"num_of_skills":df_skil["user_id"].value_counts()})


# In[61]:


temp_df


# In[62]:


arr = []
unique_vals = temp_df["num_of_skills"].unique()
for val in unique_vals:
    arr.append(temp_df[temp_df["num_of_skills"]==val].count().values[0])
    


# In[63]:


plt.figure(figsize=(13,4))
sns.barplot(x=unique_vals, y=arr);
plt.xlabel("Num of Skills",fontsize=15)
plt.ylabel("Number of People",fontsize=15)
plt.title("The Distribution of Skills by People")
plt.show()


# In[32]:


plt.figure(figsize=(12,5))
sns.barplot(x=values,y=df_skil["skill"].value_counts()[:10]);


# In[64]:


used_skills = df_skil['skill'].value_counts().iloc[:60].index

df_skil = df_skil[df_skil['skill'].isin(used_skills)]
df_skil['experience'] = True


# In[65]:


df_skil


# In[66]:


df_skil.info()


# In[67]:


df_skil = df_skil.drop_duplicates(['user_id', 'skill'])
df_skil = pd.pivot(df_skil, index='user_id', columns='skill', values='experience')


# In[68]:


df_skil = df_skil.fillna(0).astype(int)
df_skil = df_skil.drop(columns=["İngilizce"])
df_skil.head()


# In[69]:


#language;


# In[70]:


df_lang


# In[72]:


df_lang.dropna(inplace=True)


# In[73]:


fixed_language_vals = {
    "(?i).*(English|ingilizce|ingizce|inglizce|ingilizce|İngilizice|ing|eng|en|Ingles).*":"English",
    "(?i).*(turkish|türkçe|Turk|türk|turkçe|türkçe|Turkish|Turkısh).*":"Turkish",
    "(?i).*(Almanca|Deutsch|Deutsch|Deutch|almanca|German|Germany).*":"German",
    "(?i).*(Francais|French|Fransız|Fransızca|Francais|Französisch|Français).*":"French",
    "(?i).*(Italian|Italyanca|Italiano|Italien|Napolitano).*":"Italian",
    "(?i).*(Spanish|Spani|Español|İspanyolca|Espańol).*" :"Spanish",
    "(?i).*(Rusça|Russian|Russe|Russain|Rusca).*" :"Russian",
    "(?i).*(Arabic|Arapça|Arapça|Arabish).*" :"Arabic",
    "(?i).*(Japanese|Japonca|japonca|Japonca).*" :"Japanese",
}


# In[74]:


fixed_proficiency_vals = {
    'elementary': 1,
    'limited_working': 2,
    'professional_working': 3,
    'full_professional': 4,
    'native_or_bilingual': 5         
    
}


# In[75]:


df_lang.loc[:, 'proficiency'].map(fixed_proficiency_vals)


# In[76]:


df_lang["language"] = df_lang["language"].replace(fixed_language_vals,regex=True)


# In[77]:


df_lang["proficiency"] = df_lang["proficiency"].map(fixed_proficiency_vals)
df_lang


# In[78]:


df_lang=df_lang.loc[df_lang["language"].isin(fixed_language_vals.values())] 


# In[79]:


df_lang["language"].value_counts()


# In[80]:


df_lang["proficiency"].value_counts()


# In[81]:


plt.figure(figsize=(10,3))
sns.barplot(x=df_lang["proficiency"].value_counts().index,y=df_lang["proficiency"].value_counts())
plt.xlabel("Proficiencies",fontsize=14)
plt.xticks([0,1,2,3,4],["elementary",'limited_working','professional_working','full_professional','native_or_bilingual'])
plt.ylabel("Num of Proficiencies",fontsize=14)
plt.show()


# In[82]:


df_lang.drop_duplicates(["user_id","language"],inplace=True)


# In[83]:


x = np.char.array(list(df_lang["language"].value_counts().index))
y = np.array(df_lang["language"].value_counts())
percent = 100.*y/y.sum()

patches, texts = plt.pie(y,startangle=90, radius=1.2)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))

plt.legend(patches, labels, loc='upper right', bbox_to_anchor=(-0.1, 1.),
           fontsize=9)
plt.show()


# In[84]:


df_lang = pd.pivot(df_lang,index="user_id",columns="language",values="proficiency")


# In[85]:


df_lang.shape


# In[86]:


df_lang.describe().T


# In[87]:


df_lang.isna().sum()


# In[88]:


df_lang = df_lang.fillna(0)
df_lang = df_lang.astype("object")


# In[3]:


# df_work için EDA ve Önişleme


# In[89]:


df_work = df_work.drop_duplicates(["user_id","company_id"])
df_work


# In[90]:


df_work = df_work.sort_values(by=['user_id', 'start_year_month']) 


# In[91]:


temp = pd.DataFrame()

temp["data_2000_and_after"] = df_work[df_work["start_year_month"].gt(200012)].groupby("user_id").size()
temp["data_2005_and_after"] = df_work[df_work["start_year_month"].gt(200512)].groupby("user_id").size()
temp["data_2010_and_after"] = df_work[df_work["start_year_month"].gt(201012)].groupby("user_id").size()
temp["data_2015_and_after"] = df_work[df_work["start_year_month"].gt(201512)].groupby("user_id").size()
temp["data_2016_and_after"] = df_work[df_work["start_year_month"].gt(201612)].groupby("user_id").size()
temp["data_2017_and_after"] = df_work[df_work["start_year_month"].gt(201712)].groupby("user_id").size()

temp["first_exp"] = df_work.groupby("user_id")["company_id"].nth(-1).astype(str)
temp["second_exp"] = df_work.groupby("user_id")["company_id"].nth(-2).astype(str)
temp["third_exp"] = df_work.groupby("user_id")["company_id"].nth(-3).astype(str)
temp["fourth_exp"] = df_work.groupby("user_id")["company_id"].nth(-4).astype(str)

temp["first_loc"] = df_work.groupby("user_id")["location"].nth(-1).astype(str)
temp["second_loc"] = df_work.groupby("user_id")["location"].nth(-2).astype(str)
temp["third_loc"] = df_work.groupby("user_id")["location"].nth(-3).astype(str)
temp["fourth_loc"] = df_work.groupby("user_id")["location"].nth(-4).astype(str)

temp["first_entrance"] = df_work.groupby("user_id")["start_year_month"].min()
temp["last_entrance"] = df_work.groupby("user_id")["start_year_month"].max()

date_arr = []


df_work["start_year_month"] = df_work["start_year_month"].astype(str)


# In[92]:


for date_str in df_work["start_year_month"]:

    date_object = datetime.strptime(date_str, '%Y%m').date()
    date_arr.append(date_object)
    
df_work["dates"] = date_arr 
temp_date = df_work.groupby("user_id")["dates"].max() - df_work.groupby("user_id")["dates"].min()


# In[93]:


temp_date = temp_date.astype('timedelta64[D]')
temp_date = (temp_date/365).map('{:,.1f}'.format)
temp_date = pd.DataFrame({"Years of experience":temp_date})
temp_date = temp_date.astype(float)
temp[temp_date.columns] = temp_date[temp_date.columns]


# In[94]:


df_work["start_year_month"] = df_work["start_year_month"].astype(dtype="str")
df_work["start_year_month"] = (df_work["start_year_month"].str[:4]).astype("int64")


# In[95]:


sns.lineplot(x=df_work["start_year_month"],
             y=df_work["user_id"].value_counts(),
            estimator="mean")
plt.title("The Distribution of Experience as Average")
plt.xlabel("Years",fontsize=15)
plt.ylabel("Mean of Experience",fontsize=15)
plt.show();


# In[96]:


df_work.drop(columns=["location","start_year_month","company_id","dates"],inplace=True)


# In[97]:


df_work = df_work.groupby(by="user_id").count()


# In[98]:


df_work = pd.concat([df_work,temp],axis=1)
df_work


# In[99]:


df_work.describe(include="object").T


# In[100]:


df_work.describe().T


# In[101]:


df_work["data_2000_and_after"]= df_work["data_2000_and_after"].fillna(df_work["data_2000_and_after"].mean())
df_work["data_2005_and_after"]= df_work["data_2005_and_after"].fillna(df_work["data_2005_and_after"].mean())
df_work["data_2010_and_after"]= df_work["data_2010_and_after"].fillna(df_work["data_2010_and_after"].mean())

df_work["data_2015_and_after"]= df_work["data_2015_and_after"].fillna(df_work["data_2015_and_after"].median())
df_work["data_2016_and_after"]= df_work["data_2016_and_after"].fillna(df_work["data_2016_and_after"].median())
df_work["data_2017_and_after"]= df_work["data_2017_and_after"].fillna(df_work["data_2017_and_after"].median())

frequent_value_first = int(df_work["first_exp"].fillna(df_work["first_exp"]).describe()[2])
frequent_value_second = int(df_work["second_exp"].fillna(df_work["second_exp"]).describe()[2])
frequent_value_third = int(df_work["third_exp"].fillna(df_work["third_exp"]).describe()[2])
frequent_value_fourth = int(df_work["fourth_exp"].fillna(df_work["fourth_exp"]).describe()[2])

df_work["first_exp"]= df_work["first_exp"].fillna(frequent_value_first)
df_work["second_exp"]= df_work["second_exp"].fillna(frequent_value_second)
df_work["third_exp"]= df_work["third_exp"].fillna(frequent_value_third)
df_work["fourth_exp"]= df_work["fourth_exp"].fillna(frequent_value_fourth)

df_work["first_loc"] = df_work["first_loc"].replace("nan","Istanbul")
df_work["first_loc"]= df_work["first_loc"].fillna("Istanbul")

df_work["second_loc"] = df_work["second_loc"].replace("nan","Istanbul, Turkey")
df_work["second_loc"]= df_work["second_loc"].fillna("Istanbul, Turkey")

df_work["third_loc"] = df_work["third_loc"].replace("nan","Istanbul, Turkey")
df_work["third_loc"]= df_work["third_loc"].fillna("Istanbul, Turkey")

df_work["fourth_loc"] = df_work["fourth_loc"].replace("nan","Istanbul, Turkey")
df_work["fourth_loc"]= df_work["fourth_loc"].fillna("Istanbul, Turkey")

df_work["Years of experience"]= df_work["Years of experience"].fillna(df_work["Years of experience"].mean())


# In[102]:


## tüm verisetleri


# In[103]:


df_train.head()


# In[104]:


abroad_countries = df_train[~df_train["location"].str.contains("Turkey|Greater",na=False)]["location"].unique()
df_train.loc[df_train["location"].isin(abroad_countries),"location"]= "abroad_countries"


# In[105]:


df_train.loc[df_train["location"].str.contains("Istanbul|İstanbul",na=False),"location"] = "Istanbul"
df_train.loc[df_train["location"].str.contains("Ankara",na=False),"location"] = "Ankara"
df_train.loc[df_train["location"].str.contains("Kocaeli",na=False),"location"] = "Kocaeli"
df_train.loc[df_train["location"].str.contains("İzmir|Izmir",na=False),"location"] = "İzmir"
df_train.loc[df_train["location"].str.contains("Mersin|İçel",na=False),"location"] = "Mersin"
df_train.loc[df_train["location"].str.contains("Sakarya",na=False),"location"] = "Sakarya"
df_train.loc[df_train["location"].str.contains("Konya",na=False),"location"] = "Konya"

abroad_countries = df_test[~df_test["location"].str.contains("Turkey|Greater",na=False)]["location"].unique()
df_test.loc[df_test["location"].isin(abroad_countries),"location"]= "abroad_countries"


# In[106]:


df_test.loc[df_test["location"].str.contains("Istanbul|İstanbul",na=False),"location"] = "Istanbul"
df_test.loc[df_test["location"].str.contains("Ankara",na=False),"location"] = "Ankara"
df_test.loc[df_test["location"].str.contains("Kocaeli",na=False),"location"] = "Kocaeli"
df_test.loc[df_test["location"].str.contains("İzmir|Izmir",na=False),"location"] = "İzmir"
df_test.loc[df_test["location"].str.contains("Mersin|İçel",na=False),"location"] = "Mersin"
df_test.loc[df_test["location"].str.contains("Sakarya",na=False),"location"] = "Sakarya"
df_test.loc[df_test["location"].str.contains("Konya",na=False),"location"] = "Konya"


# In[107]:


df_train["location"].value_counts()[:5]


# In[108]:


df_test["location"].value_counts()[:5]


# In[109]:


fixed_industry = {
    "Computer Software|Computer Games|Mobile Games":"Computer Software",
    "Information Technology and Services|Telecommunications|Internet|Computer & Network Security|Computer Networking|Information Services":"Information Services",
    "Defense & Space|Automotive|Electrical/Electronic Manufacturing|Aviation & Aerospace|Industrial Automation|Retail|Airlines/Aviation|Mechanical or Industrial Engineering|Consumer Electronics|Textiles|Chemicals|Machinery|Construction|Mining & Metals|Oil & Energy|Consumer Goods|Food & Beverages":"Industrial and Manufacturing",
    "Pharmaceuticals|Hospital & Health Care|Medical Devices":"Health",
    "Banking|Higher Education|Financial Services|Program Development|Research|Insurance|Nonprofit Organization Management|Education Management|Marketing and Advertising|E-Learning":"Business/Education"    
}


# In[110]:


df_train.loc[:"industry"] = df_train.loc[:"industry"].replace(fixed_industry,regex=True)
df_train.loc[~df_train["industry"].isin(fixed_industry.values()),"industry"]= "Other_Sectors" 


# In[111]:


df_test.loc[:"industry"] = df_test.loc[:"industry"].replace(fixed_industry,regex=True)
df_test.loc[~df_train["industry"].isin(fixed_industry.values()),"industry"]= "Other_Sectors" 


# In[112]:


df_test.set_index("user_id",inplace=True)
df_test


# In[113]:


df_train.set_index("user_id",inplace=True)
df_train


# In[114]:


# verisetleri birleştirme


# In[115]:


df_train[df_edu.columns] = df_edu[df_edu.columns]
df_train[df_lang.columns] = df_lang[df_lang.columns]
df_train[df_work.columns] = df_work[df_work.columns]
df_train[df_skil.columns] = df_skil[df_skil.columns]


df_test[df_edu.columns] = df_edu[df_edu.columns]
df_test[df_lang.columns] = df_lang[df_lang.columns]
df_test[df_work.columns] = df_work[df_work.columns]
df_test[df_skil.columns] = df_skil[df_skil.columns]


# In[116]:


original_data = pd.concat([df_train,df_test],axis=0)
original_data


# In[117]:


categoricals = df_train.select_dtypes("object")
numericals = df_train.select_dtypes(exclude="object")


# In[118]:


for col in categoricals:
    original_data[col] = original_data[col].factorize()[0]

original_data[categoricals.columns] = original_data[categoricals.columns].astype('category')
original_data[numericals.columns] = original_data[numericals.columns].fillna(-1)


# In[119]:


original_data


# In[120]:


df_train = original_data.loc[df_train.index, df_train.columns]
df_test = original_data.loc[df_test.index, df_test.columns]


# In[121]:


corr = df_train.corr()
sns.set(font_scale=0.6)
plt.figure(figsize=(15,15))
sns.heatmap(corr,cmap="coolwarm",square=True)
plt.show()


# In[174]:


# model oluşturma


# In[122]:


# Train and Test Side 


# In[123]:


from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,classification_report,confusion_matrix
from sklearn.preprocessing import scale,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier 


# In[124]:


X = df_train.drop(columns=["moved_after_2019"])
y = df_train["moved_after_2019"]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,stratify=y)


# In[125]:


# logistic reg;


# In[126]:


logistic_model = LogisticRegression(solver="lbfgs",random_state=42).fit(X_train,y_train)


# In[127]:


cross_val_score(logistic_model,X_test,y_test,scoring="accuracy",cv=5)


# In[128]:


# decision tree


# In[129]:


"""decision_tree_model = DecisionTreeClassifier(random_state=42)
params = {"max_depth":[2,3,5,8,10],"min_samples_split":[2,3,5,10,20,50]}
cv_model = GridSearchCV(decision_tree_model,cv=5,param_grid=params,n_jobs=-1).fit(X_train,y_train)
print(cv_model.best_params_)"""


# In[130]:


final_decisionTree_model = DecisionTreeClassifier(max_depth=10,min_samples_split=5,random_state=42).fit(X_train,y_train)


# In[131]:


cross_val_score(final_decisionTree_model,X_test,y_test,scoring="accuracy",cv=5)


# In[132]:


# rondom forest


# randomForest_model = RandomForestClassifier(random_state=42,class_weight='balanced_subsample')
# params = {"n_estimators":[200,300],
#              "max_features":[3,5,7],
#              "min_samples_split":[2,5,10]}
# 
# randomForest_cv_model = GridSearchCV(randomForest_model,cv=5,param_grid=params,n_jobs=-1).fit(X_train,y_train)
# print(randomForest_cv_model.best_params_)

# In[133]:


final_randomForest_model = RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',max_features=5,min_samples_split=2,random_state=42).fit(X_train,y_train)


# In[134]:


cross_val_score(final_randomForest_model,X_test,y_test,scoring="accuracy",cv=5)


# In[135]:


# gbm;


# In[136]:


"""gbm_model = GradientBoostingClassifier(random_state=42)
params = {"learning_rate":[0.1,0.01,0.05],
           "n_estimators":[100,300],
           "max_depth":[2,3]}
gbm_cv_model = GridSearchCV(gbm_model,param_grid=params,cv=5,n_jobs=-1).fit(X_train,y_train)
print(gbm_cv_model.best_params_)"""


# In[137]:


final_gbm_model = GradientBoostingClassifier(n_estimators=300,learning_rate=0.1,max_depth=3,random_state=42).fit(X_train,y_train)


# In[139]:


cross_val_score(final_gbm_model,X_test,y_test,scoring="accuracy",cv=5)


# In[140]:


# xgboost;


# In[146]:


from xgboost import XGBClassifier


# In[141]:


"""xgb_model = XGBClassifier(tree_method="gpu_hist", enable_categorical=True)
params = {"n_estimators":[100,500],
            "max_depth":[3,5],
            "learning_rate":[0.1,0.01],
            "subsample":[0.6,0.8]
         }
xgb_cv_model = GridSearchCV(xgb_model,param_grid=params,cv=3,n_jobs=-1).fit(X_train,y_train)
print(xgb_cv_model.best_params_)"""


# In[148]:


final_xgb_model = XGBClassifier(n_estimators=500,learning_rate=0.1,max_depth=5,subsample=0.8).fit(X_train,y_train)


# In[ ]:


cross_val_score(final_xgb_model,X_test,y_test,scoring="accuracy",cv=5)


# In[149]:


# model karşılaştırma;


# In[150]:


models=[final_decisionTree_model,
        final_randomForest_model,
        final_gbm_model]

result=[]
results=pd.DataFrame(columns=["Models","Accuracy"])

for model in models:
    names=model.__class__.__name__
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    result=pd.DataFrame([[names,accuracy*100]],columns=["Models","Accuracy"])
    results=results.append(result)


# In[151]:


results.sort_values(inplace=True,by="Accuracy",ascending=True)


# In[152]:


sns.barplot(x="Accuracy",y="Models",data=results)
plt.xlabel("Accuracy %",fontsize=15)
plt.ylabel("Accuracy Rate of Models",fontsize=15);
plt.title("True Classification Rate",fontsize=15)
plt.show();


# In[155]:


# en başarılı sonucu random forest verdi


# In[156]:


# rf değişken önem düzeyi


# In[154]:


plt.figure(figsize=(10,15))
importance = pd.DataFrame({"variable_scores":final_randomForest_model.feature_importances_},
                          index=X.columns).sort_values(by="variable_scores",ascending=False)

sns.barplot(data=importance,x=importance["variable_scores"],y=importance.index)
plt.title("Variable Importance Levels",fontsize=15)
plt.xlabel("Variable Importance Scores",fontsize=15)
plt.ylabel("Variables",fontsize=15)
plt.show();


# In[157]:


# roc eğrisi


# In[158]:


roc_auc=roc_auc_score(y_test,final_randomForest_model.predict(X_test))
fpr,tpr,thresholds = roc_curve(y_test,final_randomForest_model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label="AUC (area=%0.2f)"%roc_auc)
plt.plot([0,1],[0,1],"r--")
plt.xlim([-0.025,1.05])
plt.ylim([-0.025,1.08])
plt.xlabel("False Positive Rate",fontsize=15)
plt.ylabel("True Positive Rate",fontsize=15)
plt.title("Receiver operating characteristic",fontsize=15)
plt.legend(loc="lower right")
plt.show()


# In[159]:


models=[final_decisionTree_model,
        final_randomForest_model,
        final_gbm_model]
for model in models:
    predicted=model.predict(X_test)
    score = accuracy_score(y_test,predicted)
    print("Score is: %{:.1f}".format(score*100))


# In[161]:


models=[final_decisionTree_model,
        final_randomForest_model,
        final_gbm_model]
for model in models:
    predicted = final_randomForest_model.predict(df_test)
    df_sub.loc[:,"moved_after_2019"] = predicted
    print(df_sub["moved_after_2019"].value_counts() )


# In[162]:


predicted=final_randomForest_model.predict(X_test)


# In[163]:


score = accuracy_score(y_test,predicted)
score


# In[164]:


# submission yapalım;


# In[165]:


predicted = final_randomForest_model.predict(df_test)


# In[167]:


df_sub.loc[:,"moved_after_2019"] = predicted


# In[168]:


submission = df_sub.astype(int)


# In[169]:


print(submission["moved_after_2019"].value_counts())


# In[170]:


submission.to_csv('submission.csv',index=False)


# In[177]:


#rf için başarı oranını artırmaya çalışalım;


# In[175]:


final_randomForest_model


# In[178]:


models=[final_decisionTree_model,
        final_randomForest_model,
        final_gbm_model]
for model in models:
    predicted=model.predict(X_test)
    score = accuracy_score(y_test,predicted)
    print("Score is: %{:.1f}".format(score*100))


# In[183]:


get_ipython().run_line_magic('pinfo', 'final_randomForest_model')


# In[179]:


# optuna için deneyelim;


# In[180]:


import optuna
from optuna import Trial,visualization,trial
from optuna.samplers import TPESampler


# In[181]:


def return_score(param):
    model=RandomForestClassifier(**param).fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    return acc


# In[184]:


def objective(trial):
    param={
        "max_depth":trial.suggest_int("max_depth",2,15),
        "min_samples_split":trial.suggest_int("min_child_samples",2,20),
        "max_features":trial.suggest_int("max_features",2,10),
        "n_estimators":trial.suggest_int("n_estimators",100,800),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,20),
        "class_weight":trial.suggest_categorical("class_weight",['balanced','balanced_subsample']),

                

    }
    return(return_score(param))


# In[186]:


study=optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=200)


# In[187]:


final_randomForest_model


# In[275]:


rf2=RandomForestClassifier(class_weight="balanced_subsample",max_features=5,
                           n_estimators=300,random_state=42,max_depth=29).fit(X_train,y_train)


# In[276]:


models=[final_decisionTree_model,
        final_randomForest_model,
        final_gbm_model,
       rf2]
for model in models:
    predicted=model.predict(X_test)
    score = accuracy_score(y_test,predicted)
    print("Score is: %{:.1f}".format(score*100))


# In[277]:


models=[final_decisionTree_model,
        final_randomForest_model,
        final_gbm_model,
       rf2]
for model in models:
    predicted = rf2.predict(df_test)
    df_sub.loc[:,"moved_after_2019"] = predicted
    print(df_sub["moved_after_2019"].value_counts() )


# In[278]:


predicted=rf2.predict(X_test)


# In[279]:


score = accuracy_score(y_test,predicted)
score


# In[280]:


# submission yapalım;


# In[281]:


predicted = rf2.predict(df_test)


# In[282]:


df_sub.loc[:,"moved_after_2019"] = predicted


# In[283]:


submission = df_sub.astype(int)


# In[284]:


print(submission["moved_after_2019"].value_counts())


# In[285]:


submission.to_csv('submission.csv',index=False)


# In[286]:


df_train


# In[287]:


df_test


# In[288]:


X_train


# In[290]:


y_train


# In[ ]:




