from function_file import *
def kidney_risk_assesment(message):
    import openai
    import pandas as pd

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    data = pd.read_csv("kidney_disease (1).csv")
    data.head()
    columns = pd.read_csv("data_description.txt", sep="-")
    columns = columns.reset_index()
    columns.columns = ["cols", "Full_names"]
    columns.head()
    data.columns = columns["Full_names"].values
    data.head()
    print(data.dtypes)

    features = ['packed cell volume', 'white blood cell count', 'red blood cell count']
    for feature in features:
        convert_dtypes(data, feature)
    data.drop('id', axis=1, inplace=True)
    data.head()

    cat_col, num_col = extract_cat_num(data)
    for col in cat_col:
        print(' {} has {} numbers of values i.e. {}'.format(col, data[col].nunique(), data[col].unique()))
    data["diabetes mellitus"] = data['diabetes mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes'})
    data["coronary artery disease"] = data['coronary artery disease'].replace(to_replace={'\tno': 'no', '\tyes': 'yes'})
    data['class'] = data['class'].replace(to_replace={'ckd\t': 'ckd'})
    for col in cat_col:
        print(' {} has {} numbers of values i.e. {}'.format(col, data[col].nunique(), data[col].unique()))
    data.isna().sum().sort_values(ascending=False)

    data[num_col].isnull().sum()
    for f in num_col:
        random_value_imputation(data, f)

    data[cat_col].isnull().sum()
    for f in cat_col:
        random_value_imputation(data, f)
    plt.figure(figsize=(30, 20))

    for i, feature in enumerate(num_col):
        plt.subplot(5, 3, i + 1)
        data[feature].hist()
        plt.title(feature)
    plt.figure(figsize=(30, 20))
    for i, feature in enumerate(cat_col):
        plt.subplot(4, 3, i + 1)
        sns.countplot(data[feature])
    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(18, 8))
    sns.heatmap(numeric_data.corr(), annot=True)
    # To study the RBC Count with respect to different features grouped with class
    Feature_Study = 5.6
    for i in data.columns:
        if i == Feature_Study:
            print(data.groupby([i, 'class'])['red blood cell count'].agg(['count', 'mean', 'median', 'min', 'max']),
                  end='\n \n \n')
    # To study the WBC Count with respect to different features grouped with class
    Feature_Study = 9600
    for i in data.columns:
        if i == Feature_Study:
            print(data.groupby([i, 'class'])['white blood cell count'].agg(['count', 'mean', 'median', 'min', 'max']),
                  end='\n \n \n')
    from tabulate import tabulate
    g = [['Name', 'count', 'mean', 'median', 'min', 'max']]
    for i in data.columns:
        if data[i].dtypes == 'float64':
            d = [i, data[i].count(), data[i].mean(), data[i].median(), data[i].min(), data[i].max()]
            g.append(d)
    print(tabulate(g, headers='firstrow'))
    messages = "You give detailed risk assesment for kidney diseases, based on pateint data, you also give precautions, current state, and what to do now"
    # Prompting user input for medical data criteria

    # Initialize an empty string to store user inputs


    # Outputting the collected data

    import google.generativeai as genai

    # library = input("Which Library do ou want to understand")
    genai.configure(api_key="AIzaSyArvzUuVG-TxNqeflFknBL1JlHfa5Y2Kww")

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(
        f'{messages}{message}')
    assesment = response.text



    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in cat_col:
        data[col] = le.fit_transform(data[col])
    data.head()
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    ind_col = [col for col in data.columns if col != 'class']
    dep_col = 'class'
    x = data[ind_col]
    y = data[dep_col]
    ordered_rank_feature = SelectKBest(score_func=chi2, k=20)
    ordered_feature = ordered_rank_feature.fit(x, y)
    print(ordered_feature.scores_)
    data_score = pd.DataFrame(ordered_feature.scores_, columns=['score'])
    data_x = pd.DataFrame(x.columns, columns=['feature'])
    features_rank = pd.concat([data_x, data_score], axis=1)
    features_rank.head()
    features_rank['score'].max()
    features_rank.nlargest(10, 'score')
    selected_columns = features_rank.nlargest(10, 'score')['feature'].values
    print(selected_columns)
    x_new = data[selected_columns]
    x_new.head()
    len(x_new)
    print(x_new.shape)
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x_new, y, random_state=0, test_size=0.25)
    ytrain.value_counts()  # Balanced Data
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    param = {'learning_rate': [0.05, 0.20, 0.25],
             'max_depth': [5, 8, 10],
             'min_child_weight': [1, 3, 5, 7],
             'gamma': [0.0, 0.1, 0.2, 0.4],
             'colsample_bytree': [0.3, 0.4, 0.7]}
    from sklearn.model_selection import RandomizedSearchCV
    random_search = RandomizedSearchCV(classifier, param_distributions=param, n_iter=5, scoring='roc_auc', n_jobs=-1,
                                       cv=5, verbose=3)
    random_search.fit(xtrain, ytrain)
    print(random_search.best_estimator_)
    classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                               colsample_bynode=1, colsample_bytree=0.4, gamma=0.4, gpu_id=-1,
                               importance_type='gain', interaction_constraints='',
                               learning_rate=0.05, max_delta_step=0, max_depth=10,
                               min_child_weight=1, monotone_constraints='()',
                               n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
                               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                               tree_method='exact', validate_parameters=1, verbosity=None)
    classifier.fit(xtrain, ytrain)
    return assesment


