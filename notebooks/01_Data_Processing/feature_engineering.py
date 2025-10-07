# Cell: Feature Engineering
import pandas as pd
import numpy as np
import os

# Assume train_final and test_final DataFrames are already loaded in the Colab environment
# If not, uncomment and adjust the following lines:
# train_final = pd.read_csv('/content/drive/MyDrive/Extro_Intro/data/train_final.csv')
# test_final = pd.read_csv('/content/drive/MyDrive/Extro_Intro/data/test_final.csv')

# Example new features:
# 1. Social activity score: sum of social features
social_features = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
train_final['social_activity_score'] = train_final[social_features].sum(axis=1)
test_final['social_activity_score'] = test_final[social_features].sum(axis=1)

# 2. Introversion score: combination of alone time and drained after socializing
# (Assume Drained_after_socializing is already numeric; if not, ensure it is)
introversion_features = ['Time_spent_Alone', 'Drained_after_socializing']
train_final['introversion_score'] = train_final[introversion_features].sum(axis=1)
test_final['introversion_score'] = test_final[introversion_features].sum(axis=1)

# 3. Interaction term: Time spent alone * Social event attendance
train_final['alone_x_event'] = train_final['Time_spent_Alone'] * train_final['Social_event_attendance']
test_final['alone_x_event'] = test_final['Time_spent_Alone'] * test_final['Social_event_attendance']

# 4. Ratio: Friends circle size / (Post frequency + 1)
train_final['friends_per_post'] = train_final['Friends_circle_size'] / (train_final['Post_frequency'] + 1)
test_final['friends_per_post'] = test_final['Friends_circle_size'] / (test_final['Post_frequency'] + 1)

# 5. Composite socializing score: (Social_event_attendance + Going_outside + Friends_circle_size + Post_frequency) - (Time_spent_Alone + Drained_after_socializing)
train_final['socializing_score'] = (
    train_final['Social_event_attendance'] + train_final['Going_outside'] + train_final['Friends_circle_size'] + train_final['Post_frequency']
    - train_final['Time_spent_Alone'] - train_final['Drained_after_socializing']
)
test_final['socializing_score'] = (
    test_final['Social_event_attendance'] + test_final['Going_outside'] + test_final['Friends_circle_size'] + test_final['Post_frequency']
    - test_final['Time_spent_Alone'] - test_final['Drained_after_socializing']
)

# Save engineered data
output_dir = '/content/drive/MyDrive/Extro_Intro/data'
train_final.to_csv(os.path.join(output_dir, 'train_engineered.csv'), index=False)
test_final.to_csv(os.path.join(output_dir, 'test_engineered.csv'), index=False)
print('Feature engineered train and test data saved.') 