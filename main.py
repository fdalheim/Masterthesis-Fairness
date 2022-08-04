from folktables import ACSDataSource, ACSEmployment
import pandas as pd
import matplotlib.pyplot as plt


data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA"], download=True)
features, label, group = ACSEmployment.df_to_numpy(acs_data)

features = [
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
test = acs_data[features]

test['ESP'].describe()
test['ESP'].value_counts(dropna=False)

test2=test[test['AGEP'].gt(16) & test['AGEP'].lt(90)]

label2 = pd.DataFrame(label, columns=['Employed'])
label3 = label2[test['AGEP'].gt(16) & test['AGEP'].lt(90)]

label3.describe()
label3.value_counts()

test2[label3.values]['AGEP'].values

plt.hist(test2[label3.values]['AGEP'].values, bins=72, alpha=0.5, color='green')

plt.hist(test2[~label3.values]['AGEP'].values, bins=72, alpha=0.5, color='red')

