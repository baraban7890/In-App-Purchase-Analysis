from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, r2_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def split_data(df):
    X = df.drop(columns = 'InGamePurchases')
    y = df['InGamePurchases']
    return X,y

def apply_resampling(X, y, method, random_state=7):
    '''
        This method takes in features and target dataframes (or series for target)
        and applies the sampling method passed in to 'method', for example SMOTE
    '''
    sampler = method(random_state=random_state)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[y.name] = y_resampled
    return resampled_df

#'Does standard scaler and returns the dataframe'
def standard_scale(X):
    '''
        This method does standard scale (mean 0 variance 1) and returns the scaled dataframe
        Should only be for feature dataframe (X)
    '''
    sc = StandardScaler()
    scaled_data = sc.fit_transform(X)
    scaled_df = pd.DataFrame(scaled_data,columns=X.columns)
    return scaled_df

def do_pca(X,y,num_components=7):
    '''
        This method does PCA with a specified number of components.
        Returns a dataframe with the PCA weights, the model, and the weights themselves in that order.
    '''
    pca = PCA(n_components=num_components)
    components = pca.fit_transform(X,y)
    component_range = range(1,num_components+1)
    cols = []
    for i in component_range:
         cols.append('PC'+str(i))
    pca_df = pd.DataFrame(components, columns=cols)
    pca_component_weights = pd.DataFrame(pca.components_.T, columns=cols, index=X.columns)
    return pca_df, pca, pca_component_weights



def preprocess_and_clean(gaming_df, smote=False, smoteenn=False, rus = False, ros = False, cc = False, pca = False):
    '''
        This method preprocesses the data by first ordinally encoding the Gamedifficulty and EngagementLevel columns.
        Then it one hot encodes the columns Gender, Location, and GameGenre.
        Then it splits it up into the features (X) and the target (y) which is InAppPurchases.
        Depending on which sampler is "True", it will apply that to the dataframe.
    '''
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


    X,y = split_data(gaming_df)

    if smote:
        if smoteenn or rus or ros or cc:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = apply_resampling(X,y,SMOTE)
    elif smoteenn:
        if smote or rus or ros or cc:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = apply_resampling(X,y,SMOTEENN)
    elif ros:
        if smote or rus or smoteenn or cc:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = apply_resampling(X,y,RandomOverSampler)
    elif rus:
        if smoteenn or smote or ros or cc:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = apply_resampling(X,y,RandomUnderSampler)
    elif cc:
        if smoteenn or rus or ros or smote:
            raise ValueError("Only one of smote, smoteen, rus, ros, or cc can be True at the same time.")
        else:
            gaming_df = apply_resampling(X,y, ClusterCentroids)
    if pca: ### TO DO
        pca_df = do_pca(X,y)
        gaming_df = standard_scale(pca_df)
    return gaming_df


def create_and_fit_model(model,X_train,y_train,num=10,random_state=7,max_iter=1000, kernel='rbf'):
    '''
        This method creates and fits a model that is specified, with the specified parameters.
    '''
    if model is RandomForestClassifier:
        pred_model = model(random_state=random_state,n_estimators=num)
    elif model is KNeighborsClassifier:
        pred_model = model(random_state=random_state,n_neighbors=num)
    elif model is LogisticRegression:
        pred_model = model(random_state=random_state,max_iter=max_iter)
    else:
        pred_model = model(random_state=random_state, kernel=kernel)
    pred_model.fit(X_train,y_train)
    return pred_model

def eval_model(fit_model,X_test,y_test):
    '''
        This method prints out the confusion matrix, balanced accuracy score, and classification report.
    '''
    y_pred = fit_model.predict(X_test)
    print(f'confusion matrix: {confusion_matrix(y_test,y_pred)}')
    print(f'balanced accuracy score: {balanced_accuracy_score(y_test,y_pred)}')
    print(f'classification report: {classification_report(y_test,y_pred)}')