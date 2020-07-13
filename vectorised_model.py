import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


# ASSUMPTION DATA
base_mortality = pd.read_csv('/Users/joshchapman/PycharmProjects/VectorisedAnnuityModel/venv/assumptions/base_mortality.csv', index_col=['x'])
mortality_improvements = pd.read_csv('/Users/joshchapman/PycharmProjects/VectorisedAnnuityModel/venv/assumptions/mortality_improvements.csv', index_col=['Time'])
discount_rate = 0.05

# POLICY DATA
policy_data = pd.read_csv('/Users/joshchapman/PycharmProjects/VectorisedAnnuityModel/venv/policy data/annuity_policy_data.csv', index_col=['Policy ID'], parse_dates=['Annuitant Date of Birth', 'First Payment Date'])

# INPUTS
valuation_date = datetime.strptime('31/12/2019', '%d/%m/%Y')
steps = 60


# Calculate age of each policy holder as of the valuation date
policy_data['Age at Valuation'] = (valuation_date - policy_data['Annuitant Date of Birth']) // np.timedelta64(1, 'Y')

# Construct a dataframe containing the base qx values for each policy holder at each time step
age_bins = range(16, 106)
death_prob = base_mortality['qx']
values = {policy_id: pd.cut(range(policy['Age at Valuation'], policy['Age at Valuation'] + steps), age_bins, labels=death_prob) for policy_id, policy in policy_data.iterrows()}
mortality = pd.DataFrame(values, dtype=np.float64).fillna(1)

# Apply mortality improvements to the base mortality
mortality_improvement_factors = (1 - mortality_improvements['Improvement'][0:steps]) ** (mortality_improvements.index[0:steps] + 1)
improved_mortality = mortality * np.reshape(mortality_improvement_factors.values, (-1, 1))
improved_mortality.index = improved_mortality.index + 1

# Calculate the in-force probabilities, from the improved mortality, for each policy holder at each projection year
prob_in_force = (1 - improved_mortality).cumprod()

# Construct discount curve from flat discount rate
discount_curve = pd.DataFrame(index=range(1, steps + 1))
discount_curve['Value'] = (1 + discount_rate) ** - discount_curve.index

# Project EPV of payments to each annuitant at each projection year
projection = prob_in_force * discount_curve.values * policy_data['Amount of Annuity'].values

print(projection)