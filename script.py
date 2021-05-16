import os
import csv
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA

import re
from joblib import dump, load
import warnings
import plotly.express as px
# pip install plotly
# https://plotly.com/python/pca-visualization/


# ignore all warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

#lib for regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#lib for NN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

V1='2021VAERSDATA.xlsx'
V2='2021VAERSSYMPTOMS.xlsx'
V3='2021VAERSVAX.xlsx'
foldername='./rawdata'
MERGE_FILE = 'merged_out.xlsx'
PROCESSED='processed_data.xlsx'
TARGETS='target.xlsx'
FEATURES='feat.xlsx'
suffix='.xlsx'
modchoicetext="""Please choose a model:
1. Stochastic Gradient Descent Classification
2. Logistic Regression Classification
3. K Neighbors Classification
4. Neural Network Classification
(choose number): """
#SYMP_CAT='SYMPTOMS'+suffix
#ALLERGY_CAT='ALLERGIES'+suffix

catmap_dir='./category'

def plotPCA():

    df_feat = pd.read_excel(FEATURES,encoding='windows-1252')
    df_targets = pd.read_excel(TARGETS,encoding='windows-1252')

    df_feat = df_feat[["AGE_YRS","SEX","V_ADMINBY","CUR_ILL","HISTORY",
    "ALLERGIES","VAX_MANU","VAX_ROUTE", "SYMPTOM1","SYMPTOM2","SYMPTOM3",
    "SYMPTOM4","SYMPTOM5"]]
    df_targets = df_targets[["DIED","L_THREAT","ER_ED_VISIT","HOSPITAL",
    "DISABLE","RECOVD"]]

    features = []
    for index, rows in df_feat.iterrows():
        my_list = [rows.AGE_YRS, rows.SEX, rows.V_ADMINBY, rows.CUR_ILL,
        rows.HISTORY, rows.ALLERGIES, rows.VAX_MANU , rows.VAX_ROUTE,
        rows.SYMPTOM1 ,rows.SYMPTOM2 ,rows.SYMPTOM3,
        rows.SYMPTOM4 ,rows.SYMPTOM5]
        features.append(my_list)

    targets = []
    for index, rows in df_targets.iterrows():
        my_list = [rows.DIED, rows.L_THREAT, rows.ER_ED_VISIT, rows.HOSPITAL,
        rows.DISABLE, rows.RECOVD]
        targets.append(my_list)
    df_targets.to_excel(TARGETS, index=True)

    df_targets.columns = range(df_targets.shape[1])

    targetcat = ["Death", "Live-Threatening", "ER-Visit", "Hospitalization","Disabled","Recovered"]

    df_feat.columns = range(df_feat.shape[1]) # Delete headers.
    X = df_feat.iloc[:, 0:-1].values
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    for i in range(0,len(targetcat)):
        y = df_targets.iloc[:, i].values
        """
        fig = px.scatter(components, x=0, y=1, color=y)
        fig.show()
        """
        for label in set(y):
            plt.scatter(components[y==label, 0], components[y==label, 1],
            alpha=0.5,label=label)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'Features vs Target[{targetcat[i]}] PCA')
        plt.legend()
        #plt.show()
        plt.savefig(f'{targetcat[i]}.png')
        plt.clf()

    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    for i in range(0,len(targetcat)):
        y = df_targets.iloc[:, i].values
        total_var = pca.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=y,
            title=f'{targetcat[i]} Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show()



def k_neighbors_classifier(cat,x,y):
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    neigh = KNeighborsClassifier(n_neighbors=5,weights='distance',
            algorithm='ball_tree', leaf_size=30, p=3, n_jobs=4)
    neigh.fit(X_train, y_train)
    report = classification_report(y_test, neigh.predict(X_test))
    return neigh, report

def stochastic_GD_classifier(cat,x,y):
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    sto = SGDClassifier(loss="squared_loss", penalty="l2", max_iter=1000)
    sto.fit(X_train,y_train)
    #print(classification_report(y_test, sto.predict(X_test)))
    report = classification_report(y_test, sto.predict(X_test))
    return sto, report


def neural_net_train(cat,x,y):
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    LR_init_map ={"Death": [0.001,20], "Live-Threatening":[0.0001,40], "ER-Visit":[0.0001,36],
    "Hospitalization":[0.0001,33], "Disabled":[0.0001,20],"Recovered":[0.0001,35]}
    lr_init = LR_init_map[cat][0]
    layers = LR_init_map[cat][1]

    nn = MLPClassifier(solver='adam', alpha=1e-7, hidden_layer_sizes=(10, 4),
    random_state=42,learning_rate='adaptive', learning_rate_init=lr_init,
    shuffle=True, warm_start=True, max_iter=20) # batch_size=200
    nn.fit(X_train, y_train)
    loss = []

    for i in range(layers):
        nn.fit(X_train, y_train)
        loss.append(1- nn.score(X_test, y_test, sample_weight=None))

    MLPClassifier(alpha=1e-05, hidden_layer_sizes=(10, 4), random_state=42,solver='adam',learning_rate='adaptive')

    y_pred = nn.predict(X_test)

    #plt.plot(range(layers), loss, '-')
    #plt.title(f"{cat}, lr_init = {lr_init}")
    #plt.show()

    report = classification_report(y_test, nn.predict(X_test))
    return nn, report

def classification_train(cat,x,y):
    # source:
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=500,
                       multi_class='auto', n_jobs=4, penalty='l2',
                       random_state=0, solver='sag', tol=0.0001, verbose=0,
                       warm_start=False)

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    lr.score(X_test,y_test)

    report = classification_report(y_test, lr.predict(X_test))
    return lr, report

def plot_raw_data(df):

    plt.scatter(df.AGE_YRS, df.ALLERGIES, c='red', marker = 'x')
    plt.title('Age vs Allergies plot')
    plt.show()

def merge_datasheet():
        unwant = ['unknown','Unknown','None','N/A','n/a','UNK','U','N',
        'None known','NONE','No','no','No known allergies','No known','none','UN']

        df_data = pd.read_excel(f'{foldername}/{V1}',encoding='windows-1252')
        df_data = df_data[["VAERS_ID","AGE_YRS","SEX","DIED","L_THREAT",
        "ER_ED_VISIT","HOSPITAL","DISABLE","RECOVD","V_ADMINBY","CUR_ILL","HISTORY",
        "ALLERGIES"]]
        df_data = df_data.fillna(0)
        df_data.replace(unwant, 0,inplace=True)
        #changecol = ["CUR_ILL","HISTORY"]
        df_data.loc[df_data["CUR_ILL"] != 0, "CUR_ILL"] = 1
        df_data.loc[df_data["HISTORY"] != 0, "HISTORY"] = 1

        df_symp = pd.read_excel(f'{foldername}/{V2}',encoding='windows-1252')
        df_symp = df_symp[["VAERS_ID","SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"]]
        #df_symp.merge(df_symp,on="VAERS_ID",how='inner')
        df_symp = df_symp.fillna(0)
        df_symp.replace(unwant, 0,inplace=True)

        df_vax = pd.read_excel(f'{foldername}/{V3}',encoding='windows-1252')
        del df_vax["VAX_LOT"]
        # remove all non-covid19 vaccines
        df_vax = df_vax[df_vax["VAX_TYPE"] == 'COVID19']
        del df_vax["VAX_TYPE"]
        df_vax = df_vax.fillna(0)
        df_vax.replace(unwant, 'none',inplace=True)

        df_data = df_data.fillna(0)
        df_symp = df_symp.fillna(0)
        df_vax = df_vax.fillna(0)
        frames = [df_data,df_symp,df_vax]

        df = df_data.merge(df_symp,on="VAERS_ID")
        df = df.merge(df_vax,on="VAERS_ID")
        df.to_excel(MERGE_FILE, index=True)

def pre_process():

    unwant = ['unknown','Unknown','None','N/A','n/a','UNK','U','N',
    'None known','NONE','No','no','No known allergies','No known','none','UN']
    df = pd.read_excel(MERGE_FILE,encoding='windows-1252')
    # change values
    df.replace(unwant, 0,inplace=True)

    # convert to Categorical values to number
    # text
    df.V_ADMINBY = pd.Categorical(df.V_ADMINBY)
    my_map = dict(enumerate(df.V_ADMINBY.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/V_ADMINBY"+suffix,index=False)
    df['V_ADMINBY'] = df.V_ADMINBY.cat.codes
    # text
    df.SEX = pd.Categorical(df.SEX)
    my_map = dict(enumerate(df.SEX.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/SEX"+suffix,index=False)
    df['SEX'] = df.SEX.cat.codes
    # text
    df.VAX_MANU = pd.Categorical(df.VAX_MANU)
    my_map = dict(enumerate(df.VAX_MANU.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/VAX_MANU"+suffix,index=False)
    df['VAX_MANU'] = df.VAX_MANU.cat.codes
    # text
    df.VAX_ROUTE = pd.Categorical(df.VAX_ROUTE)
    my_map = dict(enumerate(df.VAX_ROUTE.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/VAX_ROUTE"+suffix,index=False)
    df['VAX_ROUTE'] = df.VAX_ROUTE.cat.codes

    df.L_THREAT = pd.Categorical(df.L_THREAT)
    my_map = dict(enumerate(df.L_THREAT.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/L_THREAT"+suffix,index=False)
    df['L_THREAT'] = df.L_THREAT.cat.codes

    df.ER_ED_VISIT = pd.Categorical(df.ER_ED_VISIT)
    my_map = dict(enumerate(df.ER_ED_VISIT.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/ER_ED_VISIT"+suffix,index=False)
    df['ER_ED_VISIT'] = df.ER_ED_VISIT.cat.codes

    df.HOSPITAL = pd.Categorical(df.HOSPITAL)
    my_map = dict(enumerate(df.HOSPITAL.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/HOSPITAL"+suffix,index=False)
    df['HOSPITAL'] = df.HOSPITAL.cat.codes

    df.RECOVD = pd.Categorical(df.RECOVD)
    my_map = dict(enumerate(df.RECOVD.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/RECOVD"+suffix,index=False)
    df['RECOVD'] = df.RECOVD.cat.codes

    df.DIED = pd.Categorical(df.DIED)
    my_map = dict(enumerate(df.DIED.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/DIED"+suffix,index=False)
    df['DIED'] = df.DIED.cat.codes

    df.DISABLE = pd.Categorical(df.DISABLE)
    my_map = dict(enumerate(df.DISABLE.cat.categories))
    my_map = {v: k for k, v in my_map.items()}
    tmp_pd = pd.DataFrame(my_map,index=[0])
    tmp_pd.to_excel(catmap_dir+"/DISABLE"+suffix,index=False)
    df['DISABLE'] = df.DISABLE.cat.codes

    aller = []
    for item in df['ALLERGIES']:
        key = str(item).lower()
        aller.append(key)

    cols = ['SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3','SYMPTOM4','SYMPTOM5']
    df[cols] = df[cols].astype(str)

    symp = []
    # generate categorical list
    for cl in cols:
        for item in df[cl]:
            key = str(item).lower()
            symp.append(key)

    # extract main symptoms/allergies
    symp_count = Counter(symp)
    aller_count = Counter(aller)
    symp_count = {x: count for x, count in symp_count.items() if count >= 3}
    aller_count = {x: count for x, count in aller_count.items() if count >= 3}
    # assign count as categorical number to symptoms/allergies
    symptoms = {}
    symptoms['0'] = 0
    symptoms['others'] = 1
    catnum = 2

    for key in symp_count:
        if str(key) in symptoms:
            continue
        else:
            symptoms[str(key)] = catnum
            catnum+=1

    allergies = {}
    allergies['0'] = 0
    allergies['others'] = 1
    catnum = 2
    for key in aller_count:
        if str(key) in allergies:
            continue
        else:
            allergies[str(key)] = catnum
            catnum+=1

    # substitute to categorical number
    for i in range(0,len(df['ALLERGIES'])):
        key = str(df['ALLERGIES'][i]).lower()
        if key == '0':
            continue
        elif key in allergies:
            df.iloc[i,df.columns.get_loc('ALLERGIES')] = allergies[key]
        else:
            df.iloc[i,df.columns.get_loc('ALLERGIES')] = 1

    for cl in cols:
        for i in range(0,len(df[cl])):
            key = str(df[cl][i]).lower()
            if key == '0':
                continue
            elif key in symptoms:
                df.iloc[i,df.columns.get_loc(cl)] = symptoms[key]
            else:
                df.iloc[i,df.columns.get_loc(cl)] = 1

    # save assined catagories to excel
    allergy_pd = pd.DataFrame(allergies,index=[0])
    allergy_pd.to_excel(catmap_dir+"/ALLERGIES"+suffix,index=False)
    symptom_pd = pd.DataFrame(symptoms,index=[0])
    allergy_pd.to_excel(catmap_dir+"/SYMPTOMS"+suffix,index=False)
    df.to_excel(PROCESSED, index=True)

def modchoice_getname():
    modchoice=input(modchoicetext)
    modname=''
    if modchoice == '1':
        modname='SGDC'
    elif modchoice == '2':
        modname='LRC'
    elif modchoice == '3':
        modname='KNC'
    elif modchoice == '4':
        modname='NNC'

    return modname


def train_data():

    modname = modchoice_getname()

    df =pd.read_excel(PROCESSED,encoding='windows-1252')
    df_targets = df[["DIED","L_THREAT","ER_ED_VISIT","HOSPITAL",
    "DISABLE","RECOVD"]]

    targets = []
    target_head = df_targets.head()
    for index, rows in df_targets.iterrows():
        my_list = [rows.DIED, rows.L_THREAT, rows.ER_ED_VISIT, rows.HOSPITAL,
        rows.DISABLE, rows.RECOVD]
        targets.append(my_list)
    df_targets.to_excel(TARGETS, index=True)

    df_targets.columns = range(df_targets.shape[1]) # Delete headers.


    df_feat = df[["AGE_YRS","SEX","V_ADMINBY","CUR_ILL","HISTORY",
    "ALLERGIES","VAX_MANU","VAX_ROUTE",
    "SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"]]

    features = []
    feat_head = df_feat.head()
    for index, rows in df_feat.iterrows():
        my_list = [rows.AGE_YRS, rows.SEX, rows.V_ADMINBY, rows.CUR_ILL,
        rows.HISTORY, rows.ALLERGIES, rows.VAX_MANU , rows.VAX_ROUTE,
        rows.SYMPTOM1 ,rows.SYMPTOM2 ,rows.SYMPTOM3,
        rows.SYMPTOM4 ,rows.SYMPTOM5]
        features.append(my_list)

    df_feat.to_excel(FEATURES, index=True)
    df_feat.columns = range(df_feat.shape[1]) # Delete headers.

    X = df_feat.iloc[:, 0:-1].values
    #Y = df_targets.iloc[:, 0:-1].values
    targetcat = ["Death", "Live-Threatening", "ER-Visit", "Hospitalization","Disabled","Recovered"]

    model = {}
    report = {}
    if modname == 'SGDC':
        print("Stochastic Gradient Descent Classification")
        for i in range(0,len(targetcat)):
            y = df_targets.iloc[:, i].values
            model[targetcat[i]],report[targetcat[i]] = stochastic_GD_classifier(targetcat[i],X,y)
            dump(model, './models/SGDC.joblib')


    elif modname == 'LRC':
        print("Logistic Regression Classification")
        for i in range(0,len(targetcat)):
            y = df_targets.iloc[:, i].values
            model[targetcat[i]],report[targetcat[i]] = classification_train(targetcat[i],X,y)
            dump(model, './models/LRC.joblib')


    elif modname == 'KNC':
        print("K Neighbors Classification")
        for i in range(0,len(targetcat)):
            y = df_targets.iloc[:, i].values
            model[targetcat[i]],report[targetcat[i]] = k_neighbors_classifier(targetcat[i],X,y)
            dump(model, './models/KNC.joblib')

    elif modname == 'NNC':
        print("Neural Network Classification")
        for i in range(0,len(targetcat)):
            y = df_targets.iloc[:, i].values
            model[targetcat[i]],report[targetcat[i]] = neural_net_train(targetcat[i],X,y)
            dump(model, './models/NNC.joblib')


    print("================================")
    for cat  in report:
        print("Category = ", cat)
        print(report[cat])
        print("================================")


    return model

def predict_patient():


    filename = input("Input file name: ")

    df_in = pd.read_excel(filename, encoding='windows-1252')

    df_in = df_in[["AGE_YRS","SEX","V_ADMINBY","CUR_ILL","HISTORY","ALLERGIES",
    "VAX_MANU","VAX_ROUTE","SYMPTOM1","SYMPTOM2","SYMPTOM3","SYMPTOM4","SYMPTOM5"]]

    saved_features = ["SEX","V_ADMINBY", "ALLERGIES","VAX_MANU","VAX_ROUTE","SYMPTOMS"]

    # load category map
    featmap = {}
    for feat in saved_features:
        path = catmap_dir+"/"+feat+suffix
        featmap[feat] = pd.read_excel(path,encoding='windows-1252').to_dict('list')

    for feat in saved_features:
        for item in featmap[feat]:
            featmap[feat][item] = featmap[feat][item][0]

    # map to category number
    no_col = ['AGE_YRS', 'CUR_ILL', 'HISTORY']
    symp_name = ['SYMPTOM1','SYMPTOM2','SYMPTOM3','SYMPTOM4','SYMPTOM5']


    for column in df_in:
        #print(df_in[column])
        for i in range(0,len(df_in[column])):
            content = str(df_in[column][i])
            if column not in no_col:
                if column in symp_name:
                    for key in featmap['SYMPTOMS']:
                        keystr = str(key)
                        if content.lower() in keystr.lower():
                            df_in.iloc[i,df_in.columns.get_loc(column)] = featmap['SYMPTOMS'][key]
                else:
                    for key in featmap[column]:
                        keystr = str(key)
                        if content.lower() in keystr.lower():
                            df_in.iloc[i,df_in.columns.get_loc(column)] = featmap[column][key]
    for column in df_in:
        for i in range(0,len(df_in[column])):
            if column not in no_col:
                if column in symp_name:
                    if isinstance(df_in[column][i], str):
                        df_in.iloc[i,df_in.columns.get_loc(column)] = featmap['SYMPTOMS']['others']
                else:
                    if isinstance(df_in[column][i], str):
                        df_in.iloc[i,df_in.columns.get_loc(column)] = featmap[column]['others']


    df_in.columns = range(df_in.shape[1]) # Delete headers.

    #print(featmap)
    X_in = df_in.iloc[:, 0:-1].values
    targetcat = ["Death", "Live-Threatening", "ER-Visit", "Hospitalization","Disabled","Recovered"]
    for t in targetcat:
        #ret = model[t].predict(np.array(p).reshape(-1, 1))
        print(f"For target {t}:")
        modname = modchoice_getname()
        model = load(f'./models/{modname}.joblib')
        y_pred = model[t].predict(np.array(X_in))
        print(f"{t} predicted = {y_pred}")



def main():
    choice=input("Process data? [y/n]")
    print()
    if choice == 'y':
        merge_datasheet()
        pre_process()
    choice=input("Visualize raw data on PCA? [y/n]")
    print()
    if choice == 'y':
        plotPCA()
    # LogisticRegression
    choice=input("Train model? [y/n]")
    print()
    while choice.lower() == 'y':
        train_data()
        choice=input("Train model? [y/n]")
        print()

    choice=input("Predict Patient Outcome? [y/n]")
    print()
    while choice == 'y':
        predict_patient()
        choice=input("Predict Patient Outcome? [y/n]")
        print()

    print("Exit...")


if __name__ == '__main__':
    main()
