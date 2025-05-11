import pandas as pd
import os
from scipy.io.arff import loadarff 
from sklearn.preprocessing import LabelEncoder

def load_sine_0123_abrupto_data(): 
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/Synthetic/sine_0123_abrupto.csv"))
    X = df[['X1', 'X2']].to_dict(orient='records')
    y = df['class'].values
    return "sine", X, y

def load_sine_0123_gradual_data(): 
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/Synthetic/sine_0123_gradual.csv"))
    X = df[['X1', 'X2']].to_dict(orient='records')
    y = df['class'].values
    return "sine", X, y

def load_rt_8873985678962563_abrupto_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/Synthetic/rt_8873985678962563_abrupto.csv"))
    X = df[['X1', 'X2']].to_dict(orient='records')
    y = df['class'].values
    return "rt", X, y

def load_rt_8873985678962563_gradual_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/Synthetic/rt_8873985678962563_gradual.csv"))
    X = df[['X1', 'X2']].to_dict(orient='records')
    y = df['class'].values
    return "rt", X, y


def load_mixed_0101_abrupto_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/Synthetic/mixed_0101_abrupto.csv"))
    X = df[['X1', 'X2', 'X3', 'X4']].to_dict(orient='records')
    y = df['class'].values
    return "mixed", X, y


def load_mixed_0101_gradual_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/Synthetic/mixed_0101_gradual.csv"))
    X = df[['X1', 'X2', 'X3', 'X4']].to_dict(orient='records')
    y = df['class'].values
    return "mixed", X, y


def load_elec_data():
    raw_data = loadarff(os.path.join(os.path.dirname(__file__), "../datasets/Real/Elec.arff"))
    df = pd.DataFrame(raw_data[0])
    # df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/Real/Elec.arff"))
    X = df[['nswprice', 'vicprice', 'transfer']].to_dict(orient='records')
    y = df['class'].values
    y = [int(label.decode('utf-8')) for label in y]
    return "elec", X, y


def load_iot_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/Real/IoT_2020_b_0.01_fs.csv"))
    print(df.head())
    X = df.iloc[:, :-1].to_dict(orient='records')
    y = df.iloc[:, -1].values
    return "iot", X, y


def load_cic_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datasets/Real/cic_0.01km.csv"))
    print(df.head())
    X = df.iloc[:, :-1].to_dict(orient='records')
    y = df.iloc[:, -1].values
    return "cic", X, y


# Load the KDD Cup 1999 dataset
def load_kdd_data():
    raw_data = loadarff(os.path.join(os.path.dirname(__file__), "../datasets/Real/KDDCup.arff"))
    df = pd.DataFrame(raw_data[0])

    cat_features=[x for x in df.columns if df[x].dtype=="object"]
    le=LabelEncoder()
    for col in cat_features:
        if col in df.columns:
            i = df.columns.get_loc(col)
            df.iloc[:,i] = le.fit_transform(df.iloc[:,i].astype(str))


    

    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].apply(lambda x: 1 if x == 'normal.' else 0).values 
    # print(df.loc[:, 'logged_in'])
    # print(len(df.columns))
    # print(df.head())
    

    # label_encoders = {}

    # categorical_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    # # Convert categorical features from bytes to string and then to numeric values
    # for feature in categorical_features:
    #     label_encoders[feature] = LabelEncoder()

    # # Prepare your dataset (converting bytes to strings and then encoding)
    # X_encoded = []
    # for x in X:
    #     x_encoded = x.copy()
    #     for feature in categorical_features:
    #         x_encoded[feature] = label_encoders[feature].fit_transform([x[feature].decode('utf-8')])[0]
    #     X_encoded.append(x_encoded)

    # # Proceed with the encoded dataset
    # X = X_encoded

    # Check the format of the first few entries in X and y
    # print(f"First few samples of X: {X[:5]}")
    # print(f"First few labels of y: {y[:5]}")


    return "kdd", X, y