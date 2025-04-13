import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import streamlit as st

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


class AppBuilder :
    def __init__(self) : 
        self.model = UpliftModel()

    def create(self) :
        st.title("Conversion Attribution & Uplift Modelling Simulator")
        st.write("Use this simulator to model hypothetical changes and see \
             their potential impact on conversion rates.")
    
        # Upload data for simulation
        if st.button("Load Preloaded Data"):
            self.model.load("criteo-uplift-v2.1.csv.gz")
            st.success("Preloaded data loaded successfully.")
            st.dataframe(self.model.train_data.head())
            self.model.train()
    
    def change(self) : 
        st.sidebar.header("Changes to Simulate")
        changes = {}
        for column in self.model.features:  # Exclude target columns
            change = st.sidebar.number_input(f"Change in {column}", min_value=-5, max_value=15, value=0)
            if change != 0:
                changes[column] = change
        
        if st.sidebar.button("Simulate Changes"):
            if changes:
                # Select a random row to simulate changes on
                input_data = self.model.train_data.sample(1).copy()
                for feature, change in changes.items():
                    input_data[feature] += change
                
                result = self.model.predict(input_data, return_df=True)

                st.subheader("Results")
                st.dataframe(result)


if __name__ == "__main__" :
    print("This module is meant to be imported and not run directly")