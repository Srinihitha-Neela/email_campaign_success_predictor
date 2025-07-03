import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("train_F3fUq2S.csv")  

# Convert click_rate to binary 
df['clicked'] = (df['click_rate'] > 0.1).astype(int)

X = df.drop(columns=['click_rate', 'clicked', 'campaign_id'])
y = df['clicked']

categorical_features = ['times_of_day']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# A/B Simulation 

valid_times = df['times_of_day'].unique()
print("\nValid 'times_of_day' values for simulation:", valid_times)

ab_test_df = pd.DataFrame([
    {
        'subject_len': 50, 'body_len': 300, 'no_of_CTA': 2, 'is_discount': 1, 'is_urgency': 1,
        'is_personalised': 1, 'is_image': 1, 'is_offer': 1, 'is_product_launch': 0,
        'is_event_promo': 0, 'is_brand_promo': 0, 'is_seasonal': 0, 'times_of_day': valid_times[0],
        'day_of_week': 2, 'is_weekend': 0, 'target_audience': 2, 'sender': 1, 'category': 3, 'product': 4,
        'is_price': 1, 'is_timer': 0, 'is_quote': 0, 'is_emoticons': 1, 'mean_paragraph_len': 120, 'mean_CTA_len': 15
    },
    {
        'subject_len': 35, 'body_len': 150, 'no_of_CTA': 1, 'is_discount': 0, 'is_urgency': 1,
        'is_personalised': 0, 'is_image': 0, 'is_offer': 0, 'is_product_launch': 1,
        'is_event_promo': 1, 'is_brand_promo': 0, 'is_seasonal': 1, 'times_of_day': valid_times[0],
        'day_of_week': 5, 'is_weekend': 1, 'target_audience': 1, 'sender': 2, 'category': 1, 'product': 2,
        'is_price': 0, 'is_timer': 1, 'is_quote': 1, 'is_emoticons': 0, 'mean_paragraph_len': 80, 'mean_CTA_len': 12
    }
])


ab_probs = model.predict_proba(ab_test_df)[:, 1]

print("\nA/B Test Results:")
for i, prob in enumerate(ab_probs):
    print(f"Email Version {chr(65+i)} Predicted Success Probability: {prob:.2%}")






