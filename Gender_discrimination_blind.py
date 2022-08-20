from folktables import ACSDataSource, ACSEmployment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='xx-large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')
plt.rc('lines', linewidth='2')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)
features, label, group = ACSEmployment.df_to_numpy(acs_data)

# define names of all features
variable_names = [
    'AGEP',
    'SCHL',
    'MAR',
    'RELP',
    'DIS',
    'ESP',
    'CIT',
    'MIG',
    'MIL',
    'ANC',
    'NATIVITY',
    'DEAR',
    'DEYE',
    'DREM',
    'SEX',
    'RAC1P',
]

# convert array into dataframe
dataset = pd.DataFrame(features, columns=variable_names)
label = pd.DataFrame(label, columns=['ESR'])
label = label.astype(int)

# one-hot encoding drops first column, so make sure "white" is saved in group_frame. Should no longer be necessary
group = pd.DataFrame(group, columns=['race'])

# remove 'RELP' from dataframe as interpretation is not clear and too many categories
dataset = dataset.drop(['RELP'], axis=1)

# filtering out those with 16<age<90 -> 302640 total observations
label = label[dataset['AGEP'].gt(16) & dataset['AGEP'].lt(90)]
group = group[dataset['AGEP'].gt(16) & dataset['AGEP'].lt(90)]
dataset = dataset[dataset['AGEP'].gt(16) & dataset['AGEP'].lt(90)]

# quick summary
dataset.info()
dataset.describe()

# convert numericals to categorical variables / binning 'SCHL' and 'ESP' / Kearns
category_dict = {

    "SCHL": {
        1.0: "Did not finish high school",
        2.0: "Did not finish high school",
        3.0: "Did not finish high school",
        4.0: "Did not finish high school",
        5.0: "Did not finish high school",
        6.0: "Did not finish high school",
        7.0: "Did not finish high school",
        8.0: "Did not finish high school",
        9.0: "Did not finish high school",
        10.0: "Did not finish high school",
        11.0: "Did not finish high school",
        12.0: "Did not finish high school",
        13.0: "Did not finish high school",
        14.0: "Did not finish high school",
        15.0: "Did not finish high school",
        16.0: "Finished high school or equivalent",
        17.0: "Finished high school or equivalent",
        18.0: "Finished high school or equivalent",
        19.0: "Finished high school or equivalent",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
    'MIG': {
        1: 'Yes, same house (nonmovers)',
        2: 'No, outside US and Puerto Rico',
        3: 'No, different house in US or Puerto Rico',
    },
    'ESP': {
        0: 'Not own child of householder',
        1: 'Living with two parents, both working',
        2: 'Living with two parents, one working',
        3: 'Living with two parents, one working',
        4: 'Living with two parents, neither working',
        5: 'Living with one parent, working',
        6: 'Living with one parent, not working',
        7: 'Living with one parent, working',
        8: 'Living with one parent, not working',
    },
    'CIT': {
        1: 'Born in the U.S.',
        2: 'Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas',
        3: 'Born abroad of American parent(s)',
        4: 'U.S. citizen by naturalization',
        5: 'Not a citizen of the U.S.',
    },
    'MIL': {
        1: 'Now on active duty',
        2: 'On active duty in the past, but not now',
        3: 'Only on active duty for training in Reserves/National Guard',
        4: 'Never served in the military',
    },
    'ANC': {
        1: 'Single',
        2: 'Multiple',
        3: 'Unclassified',
        4: 'Not reported',
        8: 'Surpressed for data year 2018',
    }
}

# change the numerical values of the columns to the corresponding levels in string
for key in category_dict:
    dataset[key] = dataset[key].map(category_dict[key])

# Change coding for binary variables, so 0 = 'No', 1 = 'YES'
dummies = ['SEX', 'DIS', 'NATIVITY', 'DEAR', 'DEYE', 'DREM']
dataset[dummies] = np.where(dataset[dummies] == 2, 0, 1)

# drop gender from the training set
gender = dataset.pop('SEX')

# list that includes all categorical variables
categorical_variables = ['SCHL', 'MAR', 'ESP', 'MIG', 'CIT', 'MIL', 'ANC', 'RAC1P']

# Onehot-encoding, one column is dropped.
dataset = pd.get_dummies(dataset, columns=categorical_variables, prefix=categorical_variables, drop_first=True)

# split into training and test set (80:20 or 90:10?). Could split training set again to get a validation set
X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(dataset, label, gender,
                                                                               test_size=0.1, random_state=42)
# split one more time get 3 different sets. 1/9 to get (80:10:10 split) Train, Validation and Test
X_train, X_val, y_train, y_val, gender_train, gender_val = train_test_split(X_train, y_train, gender_train,
                                                                            test_size=1 / 9, random_state=42)

# Scaling of Features / Manually, use train values to avoid leakage
max_ = X_train.max(axis=0)
min_ = X_train.min(axis=0)

X_train = (X_train - min_) / (max_ - min_)
X_val = (X_val - min_) / (max_ - min_)
X_test = (X_test - min_) / (max_ - min_)

# save number of variables for input shape
num_features = [X_train.shape[1]]

# Blind Logistic Regression Benchmark, Accuracy = 0.7201
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.values.ravel())
predictions = model.predict(X_test)
score_log = model.score(X_test, y_test)
print(score_log)

# Trying to predict gender using remaining features, Accuracy = 0.5921 --> Features are not very predictive of the
# sensitive attribute
model_sensitive = LogisticRegression(max_iter=1000)
model_sensitive.fit(X_train, gender_train)
predictions_gender = model.predict(X_test)
score_log_gender = model.score(X_test, gender_test)
print(score_log_gender)

# Neural Network
# Define model design
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Ensure reproducible results
np.random.seed(1)
tf.random.set_seed(1)

model = keras.Sequential([
    layers.Dense(input_shape=num_features, units=16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(units=8, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])

# Add metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# Add early stopping parameters
early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

# Fit using batch size and epochs
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")

# Evaluate the model on train and test set
# The accuracy is better on the test set than on the training set since we randomly drop out 30% of the nodes in the
# training phase (regularization)
model.summary()
score_nn = model.evaluate(X_train, y_train)
print(score_nn[1])

# save predictions
y_hat = model.predict(X_test)

# round to get classes
y_hat_class = np.round(y_hat)

# Accuracy blind model
correct = (y_hat_class == y_test)
accuracy = np.count_nonzero(correct) / len(correct)  # / 30264
print(accuracy)

# Demographic Parity between Men and Women
rate_men = np.mean(y_hat_class[(gender_test == 1)])
rate_women = np.mean(y_hat_class[gender_test == 0])
print(rate_men - rate_women)

# change threshold value
x_axis = list(np.linspace(0, 0.1, 1001))
Fairness = []
Accuracies = []
for i in x_axis:
    y_hat_class[gender_test == 1] = np.round(y_hat[gender_test == 1])
    y_hat_class[gender_test == 0] = np.where(y_hat[gender_test == 0] >= 0.5 - i, 1, 0)

    rate_male = np.mean(y_hat_class[gender_test == 1])
    rate_female = (np.mean(y_hat_class[gender_test == 0]))
    Fairness.append(rate_male - rate_female)

    correct = (y_hat_class == y_test)
    accuracy = np.count_nonzero(correct) / len(correct)  # / 30264
    Accuracies.append(accuracy)

    if rate_male < rate_female:
        break

###

fig, ax = plt.subplots(nrows=1, ncols=2)

ax[0].plot(x_axis[:len(Accuracies)], Accuracies)
ax[0].set(xlabel='Intervention', ylabel='Accuracy')
ax[1].plot(x_axis[:len(Fairness)], Fairness)
ax[1].set(xlabel='Intervention', ylabel='Unfairness')

# Accuracy Values
print(f"Initial Accuracy: {Accuracies[0]}")
print(f"Accuracy after Invention: {Accuracies[-1]}")
accuracy_loss = Accuracies[0] - Accuracies[-1]
print(f"Loss in Accuracy: {accuracy_loss}")

print()

# Fairness Values
print(f"Initial Unfairness: {Fairness[0]}")
print(f"Unfairness after intervention: {Fairness[-1]}")
fairness_gain = Fairness[0] - Fairness[-1]
print(f"Gain in Fairness: {fairness_gain}")
print(f"Necessary intervention: {x_axis[len(Accuracies)-1]}")

# results differ only slighty. Aware has more accuracy with equal fairness
