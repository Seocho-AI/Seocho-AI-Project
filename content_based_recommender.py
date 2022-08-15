import pandas as pd
import numpy as np
from feature_processing import BreedsDataFeatureProcessor
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender():
    """
    A class used to select a target user to provide recommendations.
    Model will transform data, fit the data (for the specified user), than predict the top
    recommendations for the user.

    Attributes
    ----------
    breeds__data : DataFrame
        data frame of breeds rating data from dogtime
    user_survey_data : DataFrame
        data frame of user data from our web service's survey
    adopted_data : DataFrame
        data frame of adopted data from animal protect system (or each center's)
    abandoned_data : DataFrame
        data frame of abandoned dogs data from animal protect system
    target_user : str
        string of user's unique id that we're trying to provide recommendations for
    recommendations : ndarray
        numpy array of breeds indices to be recommended after using fit()
    _breed_rec_score : ndarray
        numpy array of Breeds where columns are Breed indices and
        carculated recommendation scores by cosine-similarity
    _breed_rec_valid : ndarray
        numpy array of Breeds where columns are Breed indices and
        carculated recommendation scores by Normalized Discounted Cumulative Gain (NDCG)
    _transform : BreedsDataFeatureProcessor
        class that contains the methods to transform dogtime dataset into feature
        matrices/feature vectors for model building/usage

    Methods
    -------
    fit(target_user)
        train the model using a specified target_user (id) from the user_data set.
        This function transforms all the necessary data and performs calculations
        using that trained info

    predict()
        return list of recommendations for the user sorted by top-rated breeds for th user


    """

    def __init__(self, breeds_data, user_survey_data, adopted_data, abandoned_data):
        self.breeds_data = breeds_data
        self.user_survey_data = user_survey_data
        self.adopted_data = adopted_data
        self.abandoned_data = abandoned_data

    def fit(self, target_user_id: str):
        """
        fit takes in a target_user str and transforms all relevant, breeds data to train
        the model. Recommend breeds indices will be in self. recommendations.
        To look at specific scores refer to self._breed_rec_scores

        :param target_user_id: str user_id from user_survey_data

        """

        self.target_user_id = target_user_id
        self._transform = BreedsDataFeatureProcessor(
            target_user_id=self.target_user_id,
            breeds_data=self.breeds_data,
            user_survey_data=self.user_survey_data,
            adopted_data=self.adopted_data,
            abandoned_data=self.abandoned_data
        )

        # get breeds features and ratings
        # 수정 필요한 부분
        adopted_dogs_features, adopted_dogs_ratings = \
            self._transform.transform_adopted_dogs()
        self._breed_rec_scores = cosine_similarity(adopted_dogs_features,adopted_dogs_ratings)

        self.recommendations = self._breed_rec_scores[:10, 1].astype(int)
    def predict(self):
        """
        predict will return the breeds_data DataFrame sorted by highly recommended breeds
        at the top-n.

        :return: DataFrame of breeds sorted by recommendation score descending
        """

        return self.breeds_data.iloc[self.recommendations.loc[:, "breed"]]

