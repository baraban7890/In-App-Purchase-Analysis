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

def apply_resampling(X, y, method, random_state=7):
    sampler = method(random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[y.name] = y_resampled
    return resampled_df

#'Does standard scaler and returns the dataframe'
def standard_scale(X):
    sc = StandardScaler()
    scaled_data = sc.fit_transform(X)
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
            gaming_df = apply_resampling(X,y,SMOTE)
    elif smoteen:
        if smote or rus or ros or cc:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = apply_resampling(X,y,SMOTEENN)
    elif rus:
        if smoteen or smote or ros or cc:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = apply_resampling(X,y,RandomUnderSampler)
    elif cc:
        if smoteen or rus or ros or smote:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = apply_resampling(X,y, cc)
    if pca: ### FIX THIS
        pca_df = do_pca(X,y)
        gaming_df = standard_scale(pca_df)
    return gaming_df
