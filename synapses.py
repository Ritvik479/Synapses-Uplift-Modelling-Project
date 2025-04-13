import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

class UpliftModel : 
    def __init__(self) :
        self.train_data = None
        self.features = [f'f{i}' for i in range(12)]
        self.model = None

    def load(self, file_path="criteo-uplift-v2.1.csv.gz", nrows=None) : 
        self.train_data = pd.read_csv(
            file_path,
            usecols=lambda col: col not in ["visit", "exposure"],
            nrows=nrows,
            na_filter=False,
            compression="gzip",
            on_bad_lines="skip",
            memory_map=True
        )
        return self.train_data

    def train(self) : 
        X_train = self.train_data[self.features + ['treatment']]
        y_train = self.train_data['conversion']

        self.model = HistGradientBoostingClassifier(max_iter=100, class_weight='balanced')
        self.model.fit(X_train, y_train)
    
    def predict(self, test_data=None, return_df=False) : 
        if test_data is None:
            raise ValueError("No test data provided or loaded.")
             
        # Prepare test set with t=1 and t=0
        X_test_treat = test_data[self.features].copy()
        X_test_treat['treatment'] = 1

        X_test_control = test_data[self.features].copy()
        X_test_control['treatment'] = 0

        # Predict probabilities for both scenarios
        prob_treat = self.model.predict_proba(X_test_treat)[:, 1]
        prob_control = self.model.predict_proba(X_test_control)[:, 1]

        result = pd.DataFrame({
            'prob_treated': prob_treat,
            'prob_control': prob_control,
            'uplift': prob_treat - prob_control
        })
        
        if return_df:
            return result
        else:
            print(result.head())