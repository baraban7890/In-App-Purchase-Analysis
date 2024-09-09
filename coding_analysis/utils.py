from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt

def split_data(df):
    '''
        Splits the dataframe into features and target variables.
    '''
    X = df.drop(columns = 'InGamePurchases')
    y = df['InGamePurchases']
    return X,y

def do_pca(X,num_components=7):
    '''
        This method does PCA with a specified number of components.
        Returns a dataframe with the PCA weights, the model, and the weights themselves in that order.
    '''
    pca = PCA(n_components=num_components)
    components = pca.fit_transform(X)
    component_range = range(1,num_components+1)
    cols = []
    for i in component_range:
         cols.append('PC'+str(i))
    pca_df = pd.DataFrame(components, columns=cols)
    pca_component_weights = pd.DataFrame(pca.components_.T, columns=cols, index=X.columns)
    return pca_df, pca, pca_component_weights

def preprocess_and_clean(gaming_df):
    '''
        This method preprocesses the data by first ordinally encoding the Gamedifficulty and EngagementLevel columns.
        Then it one hot encodes the columns Gender, Location, and GameGenre.
        Then it splits it up into the features (X) and the target (y) which is InAppPurchases.
        Depending on which sampler is "True", it will apply that to the dataframe.
        More than one sampler cannot be true, hence the if statements.
    '''

    custom_mapping = [
        ['Easy', 'Medium', 'Hard'],  # Custom order for 'GameDifficulty', 0 is easy, 1 is medium, 2 is hard
        ['Low', 'Medium', 'High']  # Custom order for 'EngagementLevel', 0 is low, 1 is medium, 2 is high
    ]

    ## Creates ordinalencoder and applies it to gamedifficulty and engagementlevel
    oe_encoder = OrdinalEncoder(categories=custom_mapping)

    encodings = oe_encoder.fit_transform(gaming_df[['GameDifficulty','EngagementLevel']])

    gaming_df[['GameDifficulty','EngagementLevel']] = encodings

    ## Creates one hot encoder and applies it to gender, location, gamegenre
    ohe = OneHotEncoder(sparse_output=False,dtype='int')

    ohe_df = pd.DataFrame(data=ohe.fit_transform(gaming_df[['Gender','Location','GameGenre']]), columns=ohe.get_feature_names_out())

    gaming_df = pd.concat([gaming_df, ohe_df], axis=1)
    ### We drop all of the columns that are one hot encoded
    ### We also drop one column from each of the one hot encoded columns in order to not have multicollinearity which may negatively affect the model.
    gaming_df = gaming_df.drop(columns = ['PlayerID','PlayTimeHours','GameGenre','Location','Gender','Gender_Male','Location_Other','GameGenre_Sports'])

    return gaming_df

def eval_model(fit_model,X_test,y_test):
    '''
        This method prints out the confusion matrix, balanced accuracy score, and classification report.
    '''
    y_pred = fit_model.predict(X_test)
    print(f'Accuracy Score: {accuracy_score(y_test,y_pred)}')
    print(f'Confusion Matrix:')
    print(confusion_matrix(y_test,y_pred))
    print(f'Balanced Accuracy Score: {balanced_accuracy_score(y_test,y_pred)}')
    print(f'Classification Report:')
    print(classification_report(y_test,y_pred))


def merge_with_dummy(model_dict, dummy_dict):
    '''
        This method merges a dictionary containing accuracies with the dummy dictionary.
    '''
    merged_dict = {**model_dict, **dummy_dict}  # Merging both dictionaries
    return merged_dict

def plot_accuracies_and_save(merged_dict, method,test_distribution, fileName):
    keys = list(merged_dict.keys())
    values = list(merged_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(keys, values, color='skyblue')  # Create the bar graph
        # Add labels and title

    plt.xlabel('Method used')
    plt.ylabel('Accuracy')
    plt.title(f'Comparing {method} with different preprocessing methods against dummy methods')

    ### 50% accuracy line
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='0.5 Accuracy')
    ### Proportions of 0's in the test data.
    plt.axhline(y=test_distribution, color='blue', linestyle='--', linewidth=2, label='0.5 Accuracy')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(fileName)
    # Display the plot
    plt.show()