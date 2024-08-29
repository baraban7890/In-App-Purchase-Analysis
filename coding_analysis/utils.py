from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

#'Does smote and returns the dataframe'
def do_smote(X,y,random_state=7):
    smote = SMOTE(random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X, y)
    smote_df = pd.DataFrame(X_smote, columns=X.columns)
    smote_df[y.name] = y_smote
    return smote_df

#'Does smoteen and returns the dataframe'

def do_smoteen(X,y,random_state=7):
    smoteen = SMOTEENN(random_state=random_state)
    X_smoteen, y_smoteen = smoteen.fit_resample(X, y)
    smoteen_df = pd.DataFrame(X_smoteen, columns=X.columns)
    smoteen_df[y.name] = y_smoteen
    return smoteen_df

#'Does random undersampler and returns the dataframe'
def do_rus(X,y,random_state=7):
    rus = RandomOverSampler(random_state=random_state)
    X_rus, y_rus = rus.fit_resample(X,y)
    rus_df = pd.DataFrame(X_rus,columns=X.columns)
    rus_df[y.name] = y_rus
    return rus_df

#'Does random oversampler and returns the dataframe'
def do_ros(X,y,random_state=7):
    ros = RandomUnderSampler(random_state=random_state)
    X_ros, y_ros = ros.fit_resample(X,y)
    ros_df = pd.DataFrame(X_ros,columns=X.columns)
    ros_df[y.name] = y_ros
    return ros_df

#'Does random cluster centroids and returns the dataframe'
def do_cc(X,y,random_state=7):
    cc = ClusterCentroids(random_state=random_state)
    X_cc, y_cc = ros.fit_resample(X,y)
    cc_df = pd.DataFrame(X_cc,columns=X.columns)
    cc_df[y.name] = y_cc
    return cc_df

#'Does standard scaler and returns the dataframe'
def standard_scale(X,y):
    sc = StandardScaler()
    scaled_data = sc.fit_transform(X,y)
    scaled_df = pd.DataFrame(scaled_data,columns=X.columns)
    return scaled_df

def do_pca(X,y,num_components=7):
    pca = PCA(n_components=num_components)
    components = pca.fit_transform(X,y)
    component_range = range(1,num_components+1)
    cols = []
    for i in component_range:
         cols.append('PC'+str(i))
    pca_df = pd.DataFrame(components, columns=cols)
    pca_component_weights = pd.DataFrame(pca.components_.T, columns=cols, index=X.columns)
    return pca_df, pca, pca_component_weights



def preprocess_and_clean(gaming_df, smote=False, smoteen=False, rus = False, ros = False, cc = False, pca = False):
    custom_mapping = [
        ['Easy', 'Medium', 'Hard'],  # Custom order for 'GameDifficulty', 0 is easy, 1 is medium, 2 is hard
        ['Low', 'Medium', 'High']  # Custom order for 'EngagementLevel', 0 is low, 1 is medium, 2 is high
    ]

    oe_encoder = OrdinalEncoder(categories=custom_mapping)

    encodings = oe_encoder.fit_transform(gaming_df[['GameDifficulty','EngagementLevel']])

    gaming_df[['GameDifficulty','EngagementLevel']] = encodings



    ohe = OneHotEncoder(sparse_output=False,dtype='int')

    ohe_df = pd.DataFrame(data=ohe.fit_transform(gaming_df[['Gender','Location','GameGenre']]), columns=ohe.get_feature_names_out())

    gaming_df = pd.concat([gaming_df, ohe_df], axis=1)
    gaming_df = gaming_df.drop(columns = ['PlayerID','PlayTimeHours','GameGenre','Location','Gender'])

    X = gaming_df.drop(columns = 'InGamePurchases')
    y = gaming_df['InGamePurchases']

    if smote:
        if smoteen or rus or ros or cc:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = do_smote(X,y)
    elif smoteen:
        if smote or rus or ros or cc:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = do_smoteen(X,y)
    elif rus:
        if smoteen or smote or ros or cc:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = do_rus(X,y)
    elif cc:
        if smoteen or rus or ros or smote:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = do_cc(X,y)
    if pca: ### FIX THIS
        pca_df = do_pca(X,y)
        gaming_df = standard_scale(pca_df,y)
        return gaming_df
    return gaming_df