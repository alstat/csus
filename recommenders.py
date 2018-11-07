from algos import cf

import implicit
import pandas as pd
import scipy.sparse as sparse

class CF(object):

    def __init__(self, df: pd.DataFrame):
        self.data = df.dropna()
        self.item_lookup = cf.lookup(cf.add_id(df))

    def prepdata(self) -> pd.DataFrame:
        """
        prepares the data
        """
        df = cf.prepdata_helper(cf.add_id(self.data))
        self.preped_data = df
        
        return df
        
    def fit(self, alpha: int = 15) -> implicit.als.AlternatingLeastSquares:
        """
        fits the model
        """
        fitted = cf.fit_helper(self.preped_data, alpha)
        self.fitted = fitted

        return fitted

    def top(self, id: int = 19, n: int = 5) -> pd.DataFrame:
        """
        computes top n similar products to id (product id)
        """
        return cf.top_helper(self.fitted, self.data, self.item_lookup, id, n)

    def recommend(self, user: int, num_items: int = 20) -> pd.DataFrame:
        """
        recommends num_items products to user
        """
        sparse_user_item = cf.sparse_matrix(self.preped_data, "CUSTOMER_ID", "PRODUCT_ID")

        user_vecs = sparse.csr_matrix(self.fitted.user_factors)
        item_vecs = sparse.csr_matrix(self.fitted.item_factors)

        return cf.recommend_helper(self.preped_data, self.item_lookup, user, sparse_user_item, user_vecs, item_vecs, num_items)