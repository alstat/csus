import implicit
import numpy as np
import pandas as pd
import random
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

def add_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ID for Product and Customer
    """
    df.loc[:, "PRODUCT_ID"] = df["TRAN_DESC"].astype("category").cat.codes
    df.loc[:, "CUSTOMER_ID"] = df["PANALOKARD_NO"].astype("category").cat.codes    

    return df

def sparse_matrix(df: pd.DataFrame, col1: str, col2: str) -> sparse.csr.csr_matrix:
    """
    Create Sparse Matrix
    """
    return sparse.csr_matrix((df["COUNT"].astype(float), (df[col1], df[col2])))

def lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    item = df[["TRAN_DESC", "PRODUCT_ID"]].drop_duplicates()
    item["PRODUCT_ID"] = item["PRODUCT_ID"].astype(str)

    return item

def prepdata_helper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select CUSTOMER_ID, PRODUCT_ID, AND PRODUC_COUNT
    """
    df = filter_helper(df)
    df = df[["PANALOKARD_NO", "TRAN_DESC", "PRODUCT_ID", "CUSTOMER_ID"]]
    df = pd.DataFrame(df.groupby(["CUSTOMER_ID", "PRODUCT_ID"])["TRAN_DESC"].count())
    df.columns = ["COUNT"]
    df = df.applymap(int).reset_index()

    return df
    
def filter_helper(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[df["TRAN_DESC"].str.contains("CORRECTION") == False, :]
    df = df.loc[df["TRAN_DESC"] != "2GO TRAVEL REBOOK", :]
    df = df.loc[df["TRAN_DESC"] != "2GO TRAVEL REBOOK MANUAL", :]
    df = df.loc[df["TRAN_DESC"] != "2GO TRAVEL REFUND", :]
    df = df.loc[df["TRAN_DESC"] != "SUN PANALO COOL-OFF", :]
    df = df.loc[df["TRAN_DESC"] != "TRUE PORTAL REBOOK MANUAL", :]
    df = df.loc[df["TRAN_DESC"] != "UCS OTHER COLLECTIONS", :]
    df = df.loc[df["TRAN_DESC"] != "Panalo Wallet Credit via Expense Account", :]

    return df

def fit_helper(df: pd.DataFrame, alpha: int = 15) -> implicit.als.AlternatingLeastSquares:
    """
    Train using Implicit Alternating Least Square
    """
    # Make a list that will be used for the sparse matrix R
    customers = list(np.sort(df["CUSTOMER_ID"].unique()))
    products = list(np.sort(df["PRODUCT_ID"].unique()))
    count = list(df.COUNT)

    rows = df["CUSTOMER_ID"].astype(int)
    cols = df["PRODUCT_ID"].astype(int)

    # Create a sparse matrix for Users and Products. This will be the R matrix from paper
    data_sparse = sparse.csr_matrix((count, (rows, cols)), shape = (len(customers), len(products)))

    # Implicit function by BenFred
    model = implicit.als.AlternatingLeastSquares(factors = 20, regularization = 0.1, iterations = 20)

    # Calculate the confidence by multiplying it by our alpha value.
    sparse_item_user = sparse_matrix(df, "PRODUCT_ID", "CUSTOMER_ID")
    sparse_user_item = sparse_matrix(df, "CUSTOMER_ID", "PRODUCT_ID")
    
    data_conf = (sparse_item_user * alpha).astype('double')
    
    # Fit the model
    model.fit(data_conf)
    
    return model

def top_helper(model: implicit.als.AlternatingLeastSquares, df: pd.DataFrame, item_lookup: pd.DataFrame, id: int = 21, n: int = 5) -> pd.DataFrame:
    """
    returns top 5 similar products to id

    # Arguments
        model: fitted implicit model to the data
        id   : item id
        n    : number of items for top similar products
    """
    # Get the user and item vectors from our trained model
    user_vecs = model.user_factors
    item_vecs = model.item_factors
    
    # Calculate the vector norms
    item_norms = np.sqrt((item_vecs * item_vecs).sum(axis = 1))
    
    # Calculate the similarity score, grab the top N items and
    # create a list of item-score tuples of most similar products
    scores = item_vecs.dot(item_vecs[id]) / item_norms
    
    top_idx = np.argpartition(scores, -n)[-n:]
    
    similar = sorted(zip(top_idx, scores[top_idx] / item_norms[id]), key = lambda x: -x[1])
    
    # Print the names of our most similar products
    item_label = lambda x: item_lookup["TRAN_DESC"].loc[item_lookup["PRODUCT_ID"] == str(x)].iloc[0]
    
    topn = pd.DataFrame(similar, columns = ["Item", "Score"])
    topn["Item"] = [item_label(item[0]) for item in similar]
    
    return topn

def recommend_helper(df: pd.DataFrame, item_lookup: pd.DataFrame, user_id: int, sparse_user_item: sparse.csr.csr_matrix, user_vecs: sparse.csr.csr_matrix, item_vecs: sparse.csr.csr_matrix, num_items: int = 10) -> pd.DataFrame:
    """
    The same recommendation function we used before
    """

    user_inter = sparse_user_item[user_id, :].toarray()

    user_inter = user_inter.reshape(-1) + 1
    user_inter[user_inter > 1] = 0

    rec_vector = user_vecs[user_id,:].dot(item_vecs.T).toarray()

    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_inter * rec_vector_scaled

    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    products = []; scores = []; desc = []
    for idx in item_idx:
        products.append(df["PRODUCT_ID"].loc[df["PRODUCT_ID"] == idx].iloc[0])
        scores.append(recommend_vector[idx])
        desc.append(item_lookup.loc[item_lookup["PRODUCT_ID"] == str(idx)]['TRAN_DESC'].iloc[0])

    return pd.DataFrame({'Product': desc, 'score': scores})