#!/usr/bin/env python
# coding: utf-8

# # Health Survey Analysis

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn.inspection import permutation_importance

import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from hyperopt import fmin, hp, tpe


# ## Importing and Joining data

# In[2]:


labs = pd.read_csv('../input/national-health-and-nutrition-examination-survey/labs.csv')
exam = pd.read_csv('../input/national-health-and-nutrition-examination-survey/examination.csv')
demo = pd.read_csv('../input/national-health-and-nutrition-examination-survey/demographic.csv')
diet = pd.read_csv('../input/national-health-and-nutrition-examination-survey/diet.csv')
ques = pd.read_csv('../input/national-health-and-nutrition-examination-survey/questionnaire.csv')

exam.drop(['SEQN'], axis = 1, inplace=True)
demo.drop(['SEQN'], axis = 1, inplace=True)
diet.drop(['SEQN'], axis = 1, inplace=True)
ques.drop(['SEQN'], axis = 1, inplace=True)

df = pd.concat([labs, exam], axis=1, join='inner')
df = pd.concat([df, demo], axis=1, join='inner')
df = pd.concat([df, diet], axis=1, join='inner')
df = pd.concat([df, ques], axis=1, join='inner')


# In[3]:


# SI columns are duplicate columns that give unit conversions
cols = [c for c in df.columns if c[-2:] != 'SI']
df = df[cols]


# In[4]:


# Removing highly specific columns, HPV type, specific tooth missing, sample weights, metadata
drop_cols = ['ORXGH', 'ORXGL', 'ORXH06', 'ORXH11', 'ORXH16', 'ORXH18', 'ORXH26', 'ORXH31', 'ORXH33', 'ORXH35', 'ORXH39', 'ORXH40', 'ORXH42', 'ORXH45', 'ORXH51', 'ORXH52', 'ORXH53', 
             'ORXH54', 'ORXH55', 'ORXH56', 'ORXH58', 'ORXH59', 'ORXH61', 'ORXH62', 'ORXH64', 'ORXH66', 'ORXH67', 'ORXH68', 'ORXH69', 'ORXH70', 'ORXH71', 'ORXH72', 'ORXH73', 
             'ORXH81', 'ORXH82', 'ORXH83', 'ORXH84', 'ORXHPC', 'ORXHPI','OHX01TC', 'OHX02TC', 'OHX03TC', 'OHX04TC', 'OHX05TC', 'OHX06TC', 'OHX07TC', 'OHX08TC', 'OHX09TC', 
             'OHX10TC', 'OHX11TC', 'OHX12TC', 'OHX13TC', 'OHX14TC', 'OHX15TC', 'OHX16TC', 'OHX17TC', 'OHX18TC', 'OHX19TC', 'OHX20TC', 'OHX21TC', 'OHX22TC', 'OHX23TC', 'OHX24TC', 
             'OHX25TC', 'OHX26TC', 'OHX27TC', 'OHX28TC', 'OHX29TC', 'OHX30TC', 'OHX31TC', 'OHX32TC', 'OHX02CTC', 'OHX03CTC', 'OHX04CTC', 'OHX05CTC', 'OHX06CTC', 'OHX07CTC', 
             'OHX08CTC', 'OHX09CTC', 'OHX10CTC', 'OHX11CTC', 'OHX12CTC', 'OHX13CTC', 'OHX14CTC', 'OHX15CTC', 'OHX18CTC', 'OHX19CTC', 'OHX20CTC', 'OHX21CTC', 'OHX22CTC', 
             'OHX23CTC', 'OHX24CTC', 'OHX25CTC', 'OHX26CTC', 'OHX27CTC', 'OHX28CTC', 'OHX29CTC', 'OHX30CTC', 'OHX31CTC','DMDHRAGE','WTDR2D','WTINT2YR','WTMEC2YR',
             'PHAFSTMN.x','SEQN','RIDSTATR'
            ]
df.drop(drop_cols, axis=1,inplace=True)


# ## Decoding Columns

# In[5]:


col_decoder = {
'ACD011A' : 'speak_english',
'AIALANGA' : 'speak_english2',
'ALQ101' : 'drink_alcohol',
'ALQ130' : 'alcohol_per_day',
'AUQ136' : 'ear_infections',
'BMDAVSAD' : 'saggital_abdominal_avg',
'BMXARMC' : 'arm_circum',
'BMXBMI' : 'BMI',
'BMXLEG' : 'leg_length',
'BMXSAD1' : 'saggital_abdominal_1',
'BMXSAD2' : 'saggital_abdominal_2',
'BMXWAIST' : 'waist_circum',
'BMXWT' : 'weight_kg',
'BPQ020' : 'high_bp',
'BPQ056' : 'measure_bp_home',
'BPQ059' : 'measure_bp_doctor',
'BPQ060' : 'cholesterol_checked',
'BPQ070' : 'cholesterol_checked_1y',
'BPQ080' : 'high_cholesterol',
'BPQ090D' : 'cholesterol_prescription',
'BPXDI1' : 'diastolic_bp',
'BPXML1' : 'cuff_max_inflation',
'BPXSY1' : 'blood_pressure_1',
'BPXSY2' : 'blood_pressure_2',
'BPXSY3' : 'blood_pressure_3',
'CBD070' : 'grocery_budget',
'CBD090' : 'nonfood_budget',
'CBD110' : 'food_budget',
'CBD120' : 'restaurant_budget',
'CBD130' : 'food_delivery_budget',
'CBQ505' : 'fast_food',
'CBQ535' : 'saw_nutrition_fast_food',
'CBQ545' : 'use_nutrition_fast_food',
'CBQ550' : 'eat_restaurants',
'CBQ552' : 'eat_chain_restaurants',
'CBQ580' : 'saw_nutrition_restaurant',
'CBQ590' : 'use_nutrition_restaurant',
'CBQ596' : 'saw_my_plate',
'CDQ001' : 'chest_pain_ever',
'CDQ010' : 'short_breath_stairs',
'CSQ030' : 'sensative_smell',
'CSQ100' : 'loss_of_taste',
'CSQ110' : 'taste_in_mouth',
'CSQ202' : 'dry_mouth',
'CSQ204' : 'nasal_congestion',
'CSQ210' : 'wisdom_teeth_removed',
'CSQ220' : 'tonsils_removed',
'CSQ240' : 'head_injury',
'CSQ250' : 'broken_nose',
'CSQ260' : 'sinus_infections',
'DBD100' : 'salt_frequency',
'DBD895' : 'meals_not_homemade',
'DBD900' : 'meals_fast_food',
'DBD905' : 'meals_prepackaged',
'DBD910' : 'frozen_meals_per_month',
'DBQ095Z' : 'salt_type',
'DBQ197' : 'milk_product_per_month',
'DBQ229' : 'milk_drinker',
'DBQ700' : 'healthy_diet',
'DEQ034C' : 'long_sleeve_shirt',
'DEQ034D' : 'use_sunscreen',
'DEQ038G' : 'sunburn_1y',
'DIQ010' : 'diabetes',
'DIQ050' : 'taking_insulin',
'DIQ160' : 'prediabetes',
'DIQ170' : 'diabetes_risk',
'DIQ172' : 'diabetes_concern',
'DIQ180' : 'blood_test_3y',
'DLQ010' : 'deaf',
'DLQ020' : 'blind',
'DLQ040' : 'mental_issues',
'DLQ050' : 'difficulty_walking',
'DLQ060' : 'difficulty_dressing',
'DLQ080' : 'difficulty_errands',
'DMDBORN4' : 'born_in_us2',
'DMDHHSIZ' : 'people_in_house',
'DMDHHSZB' : 'children_in_house',
'DMDHHSZE' : 'people_over_60_in_house',
'DMDHRBR4' : 'born_in_us',
'DMDHRGND' : 'gender2',
'DMDMARTL' : 'Marital_Status',
'DMDYRSUS' : 'years_in_US',
'DPQ010' : 'no_interest_2w',
'DPQ020' : 'depression',
'DPQ030' : 'trouble_sleeping_2w',
'DPQ040' : 'fatigue_2w',
'DPQ050' : 'eating_problems_2w',
'DPQ060' : 'feel_bad_2w',
'DPQ070' : 'trouble_concentrating_2w',
'DPQ080' : 'speaking_problems_2w',
'DPQ090' : 'suicidal_2w',
'DPQ100' : 'depression_difficulty',
'DR1.320Z' : 'water',
'DR1_320Z' : 'plain_water_yesterday',
'DR1_330Z' : 'tap_water_yesterday',
'DR1BWATZ' : 'bottled_water_yesterday',
'DR1HELPD' : 'interview_help',
'DR1TACAR' : 'dietary_alpha_carotene',
'DR1TALCO' : 'alcohol',
'DR1TATOC' : 'dietary_vitamin_e',
'DR1TBCAR' : 'dietary_beta_carotene',
'DR1TCAFF' : 'caffeine',
'DR1TCALC' : 'dietary_calcium',
'DR1TCARB' : 'carb',
'DR1TCHL' : 'dietary_choline',
'DR1TCHOL' : 'cholesterol',
'DR1TCOPP' : 'dietary_copper',
'DR1TCRYP' : 'dietary_beta_cryptoxanthin',
'DR1TFA' : 'dietary_folic_acid',
'DR1TFF' : 'folate_food',
'DR1TFIBE' : 'fiber',
'DR1TFOLA' : 'dietary_folate',
'DR1TIRON' : 'dietary_iron',
'DR1TKCAL' : 'calories',
'DR1TLYCO' : 'dietary_lycopene',
'DR1TLZ' : 'dietary_lutein',
'DR1TM181' : 'octadecenoic_percent',
'DR1TMAGN' : 'magnesium',
'DR1TMFAT' : 'monounsaturated_fats',
'DR1TMOIS' : 'moisture',
'DR1TNIAC' : 'dietary_niacin',
'DR1TP183' : 'octadecatrienoic_percent',
'DR1TPHOS' : 'dietary_phosphorus',
'DR1TPOTA' : 'dietary_potassium',
'DR1TPROT' : 'protein',
'DR1TRET' : 'dietary_retinol',
'DR1TS140' : 'tetradeconoic_percent',
'DR1TSELE' : 'dietary_selenium',
'DR1TSODI' : 'sodium',
'DR1TSUGR' : 'sugar',
'DR1TTFAT' : 'fat',
'DR1TTHEO' : 'dietary_theobromine',
'DR1TVARA' : 'dietary_vitamin_a',
'DR1TVB1' : 'dietary_b1',
'DR1TVB12' : 'dietary_b12',
'DR1TVB2' : 'dietary_b2',
'DR1TVB6' : 'dietary_b6',
'DR1TVC' : 'dietary_vit_c',
'DR1TVD' : 'dietary_vit_d',
'DR1TVK' : 'dietary_vit_k',
'DR1TZINC' : 'dietary_zinc',
'DRABF' : 'breast_fed',
'DRD340' : 'shellfish',
'DRD350A' : 'clams',
'DRD350B' : 'crabs',
'DRD350C' : 'crayfish',
'DRD350D' : 'lobsters',
'DRD350E' : 'mussels',
'DRD350F' : 'oysters',
'DRD350G' : 'scallops',
'DRD350H' : 'shrimp',
'DRD370A' : 'breaded_fish',
'DRD370B' : 'tuna',
'DRD370C' : 'bass',
'DRD370D' : 'catfish',
'DRD370E' : 'cod',
'DRD370F' : 'flatfish',
'DRD370G' : 'haddock',
'DRD370H' : 'mackerel',
'DRD370I' : 'perch',
'DRD370J' : 'pike',
'DRD370K' : 'pollock',
'DRD370L' : 'porgy',
'DRD370M' : 'salmon',
'DRD370N' : 'sardines',
'DRD370O' : 'sea_bass',
'DRD370P' : 'shark',
'DRD370Q' : 'swordfish',
'DRD370R' : 'trout',
'DRD370S' : 'walleye',
'DRD370T' : 'other_fish',
'DRQSDIET' : 'special_diet',
'DRQSDT1' : 'low_cal_diet',
'DRQSDT10' : 'high_protein_diet',
'DRQSDT11' : 'low_gluten_diet',
'DRQSDT12' : 'kidney_diet',
'DRQSDT2' : 'low_fat_diet',
'DRQSDT3' : 'low_salt_diet',
'DRQSDT4' : 'low_sugar_diet',
'DRQSDT5' : 'low_fiber_diet',
'DRQSDT6' : 'high_fiber_diet',
'DRQSDT7' : 'diabetic_diet',
'DRQSDT8' : 'muscle_diet',
'DRQSDT9' : 'low_carb_diet',
'DRQSDT91' : 'other_diet',
'DRQSPREP' : 'salt_used',
'DUQ200' : 'marijuana',
'DUQ370' : 'needle_drugs',
'FSD032A' : 'food_insecure',
'FSD032B' : 'not_enough_food',
'FSD032C' : 'cheap_food',
'FSD032D' : 'cheap_food_children',
'FSD032E' : 'bad_food_children',
'FSD032F' : 'low_food_children',
'FSD151' : 'emergency_food_received',
'FSDAD' : 'food_secure',
'FSDCH' : 'child_food_secure',
'FSDHH' : 'household_food_secure',
'FSQ162' : 'wic_received',
'FSQ165' : 'food_stamps',
'HEQ010' : 'hepetitis_b',
'HEQ030' : 'hepetitis_c',
'HIQ011' : 'health_insurance',
'HIQ210' : 'insurance_gap',
'HIQ270' : 'prescription_insurance',
'HOD050' : 'rooms_in_home',
'HOQ065' : 'homeowner',
'HSAQUEX' : 'health_status_source_data',
'HSD010' : 'general_health',
'HSQ500' : 'ever_had_cold',
'HSQ510' : 'intestinal_illness',
'HSQ520' : 'ever_had_flu',
'HSQ571' : 'donate_blood',
'HSQ590' : 'hiv',
'HUQ010' : 'general_health2',
'HUQ020' : 'health_compared_last_year',
'HUQ030' : 'routine_healthcare',
'HUQ041' : 'healthcare_location',
'HUQ051' : 'dr_visits',
'HUQ071' : 'overnight_hospital',
'HUQ090' : 'mental_health_treatment',
'IMQ011' : 'hepatitis_a_vaccine',
'IMQ020' : 'hepatitis_b_vaccine',
'IND235' : 'monthly_income',
'INDFMMPC' : 'poverty_level_category',
'INDFMMPI' : 'poverty_level_index',
'INDFMPIR' : 'family_income',
'INQ012' : 'self_employ_income',
'INQ020' : 'income_from_wages',
'INQ030' : 'income_from_SS',
'INQ060' : 'disability_income',
'INQ080' : 'retirement_income',
'INQ090' : 'ss_income',
'INQ132' : 'state_assistance_income',
'INQ140' : 'investment_income',
'INQ150' : 'other_income',
'INQ244' : 'family_savings',
'LBDBCDLC' : 'blood_cadmium',
'LBDBGMLC' : 'methyl_mercury',
'LBDHDD' : 'HDL_mg',
'LBDIHGLC' : 'inorganic_mercury',
'LBDNENO' : 'neutrophils_percent',
'LBDTHGLC' : 'blood_mercury',
'LBDWFL' : 'floride_water',
'LBXEOPCT' : 'eosinophils_percent',
'LBXGH' : 'glyco_hemoglobin',
'LBXLYPCT' : 'lymphocite_percent',
'LBXMC' : 'hemoglobin_concentration',
'LBXSAL' : 'blood_albumin',
'LBXSCA' : 'blood_calcium',
'LBXSGL' : 'serum_glucose_mg',
'LBXSTP' : 'blood_protein',
'MCQ010' : 'asthma_ever',
'MCQ025' : 'asthma_age',
'MCQ035' : 'asthma',
'MCQ040' : 'asthma_year',
'MCQ050' : 'asthma_ER',
'MCQ053' : 'anemia',
'MCQ070' : 'psoriasis',
'MCQ080' : 'overweight',
'MCQ082' : 'celiac_disease',
'MCQ086' : 'gluten_free',
'MCQ092' : 'blood_transfusion',
'MCQ149' : 'menstruate',
'MCQ151' : 'menstruate_age',
'MCQ160A' : 'arthritis',
'MCQ160B' : 'congestive_heart_failure',
'MCQ160C' : 'coronary_heart_disease',
'MCQ160D' : 'angina',
'MCQ160E' : 'heart_attack',
'MCQ160F' : 'stroke',
'MCQ160G' : 'emphysema',
'MCQ160K' : 'bronchitis_ever',
'MCQ160L' : 'liver_condition_ever',
'MCQ160M' : 'thyroid_ever',
'MCQ160N' : 'gout',
'MCQ160O' : 'COPD',
'MCQ170K' : 'bronchitis_now',
'MCQ170L' : 'liver_condition',
'MCQ170M' : 'thyroid_now',
'MCQ180A' : 'arthritis_age',
'MCQ180B' : 'heart_failure_age',
'MCQ180C' : 'heart_disease_age',
'MCQ180D' : 'angina_age',
'MCQ180E' : 'heart_attack_age',
'MCQ180F' : 'stroke_age',
'MCQ180G' : 'emphysema_age',
'MCQ180K' : 'bronchitis_age',
'MCQ180L' : 'liver_condition_age',
'MCQ180M' : 'thyroid_age',
'MCQ180N' : 'gout_age',
'MCQ195' : 'arthritis_type',
'MCQ203' : 'jaundice',
'MCQ206' : 'jaundice_age',
'MCQ220' : 'cancer',
'MCQ230A' : 'cancer_type1',
'MCQ230B' : 'cancer_type2',
'MCQ230C' : 'cancer_type3',
'MCQ230D' : 'cancer_type4',
'MCQ240A' : 'bladder_cancer_age',
'MCQ240AA' : 'test_cancer_age',
'MCQ240B' : 'blood_cancer_age',
'MCQ240BB' : 'thyroid_cancer_age',
'MCQ240C' : 'bone_cancer_age',
'MCQ240CC' : 'uterine_cancer_age',
'MCQ240D' : 'brain_cancer_age',
'MCQ240DK' : 'cancer_age',
'MCQ240E' : 'breast_cancer_age',
'MCQ240F' : 'cervical_cancer_age',
'MCQ240G' : 'colon_cancer_age',
'MCQ240H' : 'esoph_cancer_age',
'MCQ240I' : 'gallbladder_cancer_age',
'MCQ240J' : 'kidney_cancer_age',
'MCQ240K' : 'larynx_cancer_age',
'MCQ240L' : 'leukemia_age',
'MCQ240M' : 'liver_cancer_age',
'MCQ240N' : 'lung_cancer_age',
'MCQ240O' : 'lymphoma_age',
'MCQ240P' : 'melanoma_age',
'MCQ240Q' : 'mouth_cancer_age',
'MCQ240R' : 'nervous_cancer_age',
'MCQ240S' : 'ovarian_cancer_age',
'MCQ240T' : 'pancreatic_cancer_age',
'MCQ240U' : 'prostate_cancer_age',
'MCQ240V' : 'rectal_cancer_age',
'MCQ240X' : 'skin_cancer_age',
'MCQ240Y' : 'soft_cancer_age',
'MCQ240Z' : 'stomach_cancer_age',
'MCQ300A' : 'relative_heart_attack',
'MCQ300B' : 'relative_asthma',
'MCQ300C' : 'relative_diabetes',
'MCQ365A' : 'need_weight_loss',
'MCQ365B' : 'need_exercise',
'MCQ365C' : 'need_reduce_salt',
'MCQ365D' : 'need_reduce_calories',
'MCQ370A' : 'losing_weight',
'MCQ370B' : 'excercising',
'MCQ370C' : 'reducing_salt',
'MCQ370D' : 'reducing_fat',
'MGDCGSZ' : 'grip_strength',
'OCD150' : 'work_done',
'OCD270' : 'months_of_work',
'OCD390G' : 'type_of_work',
'OCD395' : 'job_duration',
'OCQ260' : 'non_govt_employee',
'OHQ030' : 'visit_dentist',
'OHQ033' : 'dentist_reason',
'OHQ620' : 'aching_mouth',
'OHQ640' : 'mouth_problems',
'OHQ680' : 'mouth_problems2',
'OHQ770' : 'need_dental',
'OHQ835' : 'gum_disease',
'OHQ845' : 'teeth_health',
'OHQ850' : 'gum_treatment',
'OHQ855' : 'loose_teeth',
'OHQ860' : 'teeth_bone_loss',
'OHQ865' : 'weird_tooth',
'OHQ870' : 'floss',
'OHQ875' : 'use_mouthwash',
'OHQ880' : 'oral_cancer_exam',
'OHQ885' : 'oral_cancer_exam2',
'OSQ060' : 'osteoporosis',
'OSQ130' : 'take_prednisone',
'OSQ230' : 'metal_objects',
'PAAQUEX' : 'question_source',
'PAD680' : 'sedentary_time',
'PAQ605' : 'vigorous_work',
'PAQ620' : 'moderate_work',
'PAQ635' : 'walk_or_bike',
'PAQ650' : 'vigorous_recreation',
'PAQ665' : 'moderate_recreation',
'PAQ710' : 'tv_hours',
'PAQ715' : 'pc_hours',
'PEASCST1' : 'bp_status',
'PEASCTM1' : 'blood_pressure_time',
'PFQ049' : 'work_limitations',
'PFQ051' : 'work_limitations2',
'PFQ054' : 'walk_equipment_required',
'PFQ057' : 'confusion_memory_problems',
'PFQ090' : 'special_healthcare_equipment',
'PUQ100' : 'insecticide_used',
'PUQ110' : 'weedkiller_used',
'RIAGENDR' : 'gender',
'RIDAGEYR' : 'age',
'RIDRETH1' : 'hispanic',
'RXQ510' : 'take_aspirin',
'SEQN' : 'ID',
'SLD010H' : 'sleep_hours',
'SLQ050' : 'trouble_sleeping',
'SLQ060' : 'sleep_disorder',
'SMAQUEX.x' : 'question_mode',
'SMAQUEX.y' : 'question_mode2',
'SMAQUEX2' : 'question_mode3',
'SMD460' : 'smokers_in_house',
'SMDANY' : 'tobaco_1w',
'SMQ681' : 'smoked_1w',
'SMQ851' : 'tobaco2_1w',
'SMQ856' : 'smoked_at_work',
'SMQ858' : 'someone_smoked_at_job',
'SMQ860' : 'smoked_at_restaurant',
'SMQ863' : 'nicotine_1w',
'SMQ866' : 'smoked_at_bar',
'SMQ870' : 'smoked_in_car',
'SMQ872' : 'someone_smoked_in_car',
'SMQ874' : 'smoked_another_home',
'SMQ876' : 'someone_smoked_in_home',
'SMQ878' : 'smoked_other_building',
'SMQ880' : 'someone_smoked_other_building',
'SXD021' : 'sex_ever',
'URXUCR' : 'creatinine_urine',
'WHD010' : 'height_in',
'WHD020' : 'current_weight_lb',
'WHD050' : 'weight_1y',
'WHD110' : 'weight_10y',
'WHD120' : 'weight_age_25',
'WHD140' : 'greatest_weight',
'WHQ030' : 'overweight_self',
'WHQ040' : 'weightloss_desire',
'WHQ070' : 'weightloss_attempt',
'WHQ150' : 'age_when_heaviest'
}
df = df.rename(columns = col_decoder)
labs = labs.rename(columns = col_decoder)
exam = exam.rename(columns = col_decoder)
demo = demo.rename(columns = col_decoder)
diet = diet.rename(columns = col_decoder)
ques = ques.rename(columns = col_decoder)


# In[6]:


cancer_df = df.dropna(subset=['cancer'])
diabetes_df = df.dropna(subset=['diabetes'])
heart_df = df.dropna(subset=['coronary_heart_disease'])
liver_df = df.dropna(subset=['liver_condition'])


# In[7]:


target_dfs = [cancer_df, diabetes_df, heart_df, liver_df]


# ## Handling Nulls and Category columns

# #### Combinations of 7s and 9s are used when data is not applicable or when the patient refused to answer

# In[8]:


for df in target_dfs:
    df.replace({7:None, 9:None, 77:None,99:None,777:None,999:None,7777:None,9999:None,77777:None,99999:None,
            777777:None,999999:None,55:None,555:None,5555:None,8:None,88:None}, inplace=True)


# #### Remove columns and rows with excessive nulls

# In[9]:


def filter_columns(df, cutoff=0.9):
    tot_rows = df.shape[0]
    removed_cols = []
    print("original number of columns: ", df.shape[1])
    for col in df.columns:
        num_na = df[col].isna().sum()
        if (num_na/tot_rows) > cutoff:
            #print(col, df[col].isna().sum())
            removed_cols.append(col)
    print("number of columns removed: ", len(removed_cols))
    return df.drop(removed_cols, axis=1)
    
def filter_rows(df, cutoff=0.9):
    tot_cols = df.shape[1]
    print("original number of rows: ", df.shape[0])
    df = df[df.isnull().sum(axis=1) < tot_cols*cutoff]
    print("remaining rows: ", df.shape[0])
    return df


# In[10]:


#df = df[df.age > 18]


# In[11]:


def trans_cat_cols(df, cat_cols):
    for col in cat_cols:
        df.loc[df[col] != 1, col] = 0
    return df


# In[12]:


for df in target_dfs:
    x = df.nunique()
    cat_cols = x[(x<15)].index
    df = trans_cat_cols(df, cat_cols)


# In[13]:


for df in target_dfs:
    df = filter_rows(df, cutoff=0.8)
    df = filter_columns(df, cutoff=0.5)


# In[14]:


for df in target_dfs:
    df.fillna(df.mode().iloc[0], inplace=True)


# ## Exploring Factors

# In[15]:


lifestyle_cols = ['drink_alcohol','alcohol_per_day','saggital_abdominal_avg','arm_circum','BMI',
                  'saggital_abdominal_1','saggital_abdominal_2','waist_circum','weight_kg',
                  'grocery_budget','nonfood_budget','food_budget',
                  'restaurant_budget','food_delivery_budget','fast_food','saw_nutrition_fast_food',
                  'use_nutrition_fast_food','eat_restaurants','eat_chain_restaurants','saw_nutrition_restaurant',
                  'use_nutrition_restaurant','saw_my_plate','wisdom_teeth_removed','tonsils_removed',
                  'salt_frequency','meals_not_homemade','meals_fast_food','meals_prepackaged',
                  'frozen_meals_per_month','salt_type','milk_product_per_month','milk_drinker','healthy_diet',
                  'long_sleeve_shirt','use_sunscreen','people_in_house','children_in_house','Marital_Status',
                  'trouble_sleeping_2w','eating_problems_2w','water','plain_water_yesterday','tap_water_yesterday',
                  'bottled_water_yesterday','dietary_alpha_carotene','alcohol','dietary_vitamin_e',
                  'dietary_beta_carotene','caffeine','dietary_calcium','carb','dietary_choline','cholesterol',
                  'dietary_copper','dietary_beta_cryptoxanthin','dietary_folic_acid','folate_food','fiber',
                  'dietary_folate','dietary_iron','calories','dietary_lycopene','dietary_lutein',
                  'octadecenoic_percent','magnesium','monounsaturated_fats','moisture','dietary_niacin',
                  'octadecatrienoic_percent','dietary_phosphorus','dietary_potassium','protein','dietary_retinol',
                  'tetradeconoic_percent','dietary_selenium','sodium','sugar','fat','dietary_theobromine',
                  'dietary_vitamin_a','dietary_b1','dietary_b12','dietary_b2','dietary_b6','dietary_vit_c',
                  'dietary_vit_d','dietary_vit_k','dietary_zinc','shellfish','clams','crabs','crayfish',
                  'lobsters','mussels','oysters','scallops','shrimp','breaded_fish','tuna','bass','catfish',
                  'cod','flatfish','haddock','mackerel','perch','pike','pollock','porgy','salmon','sardines',
                  'sea_bass','shark','swordfish','trout','walleye','other_fish','special_diet','low_cal_diet',
                  'high_protein_diet','low_gluten_diet','kidney_diet','low_fat_diet','low_salt_diet','low_sugar_diet',
                  'low_fiber_diet','high_fiber_diet','muscle_diet','low_carb_diet','other_diet','salt_used',
                  'marijuana','needle_drugs','food_insecure','not_enough_food','cheap_food','cheap_food_children',
                  'bad_food_children','low_food_children','emergency_food_received','food_secure','child_food_secure',
                  'household_food_secure','wic_received','food_stamps','health_insurance','insurance_gap',
                  'prescription_insurance','donate_blood','routine_healthcare','healthcare_location','dr_visits',
                  'hepatitis_a_vaccine','hepatitis_b_vaccine','neutrophils_percent','floride_water',
                  'overweight','gluten_free','losing_weight','excercising',
                  'reducing_salt','reducing_fat','work_done','months_of_work','type_of_work','job_duration',
                  'non_govt_employee','visit_dentist','floss','use_mouthwash','take_prednisone','sedentary_time',
                  'vigorous_work','moderate_work','walk_or_bike','vigorous_recreation','moderate_recreation',
                  'tv_hours','pc_hours','bp_status','insecticide_used','weedkiller_used','age',
                  'sleep_hours','smokers_in_house','tobaco_1w','smoked_1w','tobaco2_1w','smoked_at_work',
                  'someone_smoked_at_job','smoked_at_restaurant','nicotine_1w','smoked_at_bar','smoked_in_car',
                  'someone_smoked_in_car','smoked_another_home','someone_smoked_in_home','smoked_other_building',
                  'someone_smoked_other_building','weight_1y','weight_10y','weight_age_25','greatest_weight',
                  'overweight_self','weightloss_desire','weightloss_attempt']


# In[16]:


cols = ['age','BMI','calories','family_income']

fig, ax = plt.subplots(2, 2)
for i, col in enumerate(cols):
    ax[i//2, i%2].hist(df[col])
    ax[i//2, i%2].set_title(col)
    ax[i//2, i%2].set_ylabel("Num of Patients")
plt.tight_layout()


# In[17]:


targets = ['cancer','diabetes','coronary_heart_disease','liver_condition']


# In[18]:


plot = df[targets].mean().sort_values().plot.barh(figsize=(8,8))
plot.set_xlabel("Prevelance")
plot.set_title("Disease Prevelance for all groups")
plt.show()


# In[19]:


target_df = df[targets]
target_df = target_df[target_df.mean(axis=1) == 1/4]
target_df['disease'] = target_df.idxmax(1)
target_df['counter'] = 1
plt.figure(figsize = (10,10))
target_df.groupby('disease').counter.count().plot(kind='pie')


# ## Finding Correlations with Disease

# In[20]:


for df in target_dfs:
    ex_df = df[list(set(df.columns) & set(lifestyle_cols))]
    corr = ex_df.corr()
    corr = corr.mask(np.tril(np.ones(corr.shape)).astype(np.bool))
    redun = corr[abs(corr) >= 0.9].stack().reset_index()['level_1']
    ex_df = ex_df.drop(redun, axis=1)
    corr = ex_df.corr()
    corr = corr.mask(np.tril(np.ones(corr.shape)).astype(np.bool))
    big_corr = ex_df[corr[abs(corr).max() > 0.5].index].corr()
    big_corr = big_corr.mask(np.tril(np.ones(big_corr.shape)).astype(np.bool))
    big_corr = ex_df[big_corr[abs(big_corr).max() > 0.5].index].corr()

    plt.figure(figsize = (10,8))
    sns.heatmap(big_corr, 
            xticklabels=big_corr.columns,
            yticklabels=big_corr.columns,
            cmap="PiYG",
            vmin=-1, vmax=1)


# In[21]:


neg_factors = set()
pos_factors = set()
for col, df in zip(targets, target_dfs):
    ex_df = df[list(set(df.columns) & set(lifestyle_cols))]
    print('==========================')
    print("Correlations with ", col)
    print('==========================')
    corr = ex_df.corrwith(df[col])
    neg_factors.update(corr.sort_values().dropna().head(10).index)
    pos_factors.update(corr.sort_values().dropna().tail(10).index)
    print(corr.sort_values().dropna().head(5))
    print(corr.sort_values().dropna().tail(10))


# In[22]:


diabetes_df.age.hist()


# In[23]:


for col in ['age','greatest_weight']:#,'tetradeconoic_percent','BMI','restaurant_budget','sugar','protein']:
    plt.figure(figsize=(8,6))
    plt.hist(diabetes_df[diabetes_df.diabetes == 1][col], bins=10, alpha=0.5, label="diabetes", density=True)
    plt.hist(diabetes_df[diabetes_df.diabetes == 0][col], bins=10, alpha=0.5, label="healty", density=True)
    plt.xlabel(col, size=14)
    plt.ylabel("Percent", size=14)
    plt.title(col + " Distribution Comparison")
    plt.legend(loc='upper right')
    plt.show()


# In[24]:


for col in ['job_duration','age','child_food_secure','smoked_at_work','dr_visits','food_insecure','dietary_vitamin_a']:
    plt.figure(figsize=(8,6))
    plt.hist(cancer_df[cancer_df.cancer == 1][col], bins=10, alpha=0.5, label="cancer", density=True)
    plt.hist(cancer_df[cancer_df.cancer == 0][col], bins=10, alpha=0.5, label="healty", density=True)
    plt.xlabel(col, size=14)
    plt.ylabel("Percent", size=14)
    plt.title(col + " Distribution Comparison")
    plt.legend(loc='upper right')
    plt.show()


# In[25]:


neg_factors


# In[26]:


pos_factors


# ## Decision Trees

# In[27]:


def tree_diag(target, df, depth=5, ratio=3):
    ex_df = df[list(set(df.columns) & set(lifestyle_cols))]
    tree_mod = DecisionTreeClassifier(max_depth=depth, class_weight={0:1,1:ratio})
    tree_mod.fit(ex_df, df[target])
    fig = plt.figure(figsize=(18,13))
    _ = tree.plot_tree(tree_mod,
                      feature_names = ex_df.columns,
                      class_names = ['healthy',target],
                      filled=True)
def tree_f1(target, df, ratio=3):
    ex_df = df[list(set(df.columns) & set(lifestyle_cols))]
    X_train, X_test, y_train, y_test = train_test_split(ex_df, df[[target]], test_size=0.2)
    tree_mod = DecisionTreeClassifier(class_weight={0:1,1:ratio}, ccp_alpha=0.001)
    tree_mod.fit(X_train, y_train)
    return(f1_score(tree_mod.predict(X_test),y_test))


# In[28]:


print('Diabetes F1 score: ', tree_f1('diabetes', diabetes_df, ratio=2))
print('Cancer F1 score: ', tree_f1('cancer', cancer_df, ratio=5))
print('Heart Disease F1 score: ', tree_f1('coronary_heart_disease', heart_df, ratio=8))
print('Liver Disease F1 score: ', tree_f1('liver_condition', liver_df, ratio=10))


# In[29]:


tree_diag('diabetes', diabetes_df, depth=3)


# In[30]:


tree_diag('cancer', cancer_df, depth=3, ratio=5)


# In[31]:


tree_diag('coronary_heart_disease', heart_df, 3, ratio=8)


# In[32]:


tree_diag('liver_condition', liver_df, 3, ratio=1)


# ## Support function

# In[33]:


def nhanes_pred(model, target, df, proba=False):
    ex_df = df[list(set(df.columns) & set(lifestyle_cols))]
    X = ex_df
    y = df[[target]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train.values.ravel())
    if proba:
        preds = model.predict_proba(X_test)[:,1]
    else:
        preds = model.predict(X_test)
    return y_test, preds

def plot_roc(y_test, probs):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))
    plt.plot([0,1],[0,1],'r--')

    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(loc='lower right')
    plt.show()


# ## Naive Bayes

# In[34]:


fig, ax = plt.subplots(2,2, figsize=(15,15))
for (i, target), df in zip(enumerate(targets), target_dfs):
    nb = GaussianNB()#var_smoothing=0.0000001)
    y_test, preds = nhanes_pred(nb, target, df)
    print(target, ' F1 score: ', f1_score(y_test,preds))
    mat = confusion_matrix(y_test, preds)
    sns.heatmap(mat, ax=ax[i%2, i//2], square=True, annot=True, cbar=False, fmt='d').set_title(target)
    
plt.show()


# In[35]:


nb = GaussianNB()
ex_df = cancer_df[list(set(cancer_df.columns) & set(lifestyle_cols))]
X = ex_df
y = cancer_df[['cancer']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
nb.fit(X_train, y_train.values.ravel())


# In[36]:


imps = permutation_importance(nb, X_test, y_test)
importances = imps.importances_mean
#std = imps.importances_std
#indices = np.argsort(importances)[::-1]

#plt.figure(figsize=(10, 7))
#plt.title("Feature importances")
#plt.bar(range(X_test.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
#plt.xticks(range(X_test.shape[1]), [features[indices[i]] for i in range(6)])
#plt.xlim([-1, X_test.shape[1]])
#plt.show()


# In[37]:


imp_df = pd.DataFrame(ex_df.columns, columns=['feature'])
imp_df['importance'] = importances
imp_df.set_index('feature').sort_values(by='importance').tail(10).plot.barh()


# ## SVM

# In[38]:


space = {
        'C': hp.quniform('C', 0.005,1.0,0.01),
        'kernel': hp.choice('kernel', ['poly', 'sigmoid', 'rbf']),
        'degree': hp.choice('degree', [2,3,4])
        }


# In[39]:


ex_df = diabetes_df[list(set(diabetes_df.columns) & set(lifestyle_cols))]
X_train, X_test, y_train, y_test = train_test_split(ex_df, diabetes_df[['diabetes']], test_size=0.2)


# In[40]:


def svm_score(params):
    mod = svm.SVC(**params)
    mod.fit(X_train, y_train.values.ravel())
    predictions = mod.predict(X_test)
    return 1 - f1_score(y_test, predictions)


# In[41]:


def get_params(score):    
    best = fmin(score, space, algo=tpe.suggest, max_evals=100)
    return best


# In[42]:


get_params(svm_score)


# In[43]:


best_params = {'C': 0.92, 'degree': 2, 'kernel': 'sigmoid'}


# In[44]:


for target, df in zip(targets, target_dfs):
    mod = svm.SVC(**best_params)
    ex_df = df[list(set(df.columns) & set(lifestyle_cols))]
    X_train, X_test, y_train, y_test = train_test_split(ex_df, df[[target]], test_size=0.2)
    mod.fit(X_train, y_train.values.ravel())
    predictions = mod.predict(X_test)
    print('F1 score for ', target, ': ', f1_score(y_test, predictions))


# ## XGBoost

# ### Results are poor when using default parameters. 
# ### Below is a systematic search of hyperparamters

# In[45]:


#Dataset is imbalanced, so we will pass a balancing parameter to XGBoost
cancer_df.cancer.value_counts()[0]/cancer_df.cancer.value_counts()[1]


# In[46]:


space = {
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 8, 1),
        'subsample': hp.quniform('subsample', 0.7, 1, 0.1),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.1),
        'scale_pos_weight': 10,
        'eval_metric': 'auc'
    }


# In[47]:


def xg_score(params):
    mod = xgb.train(params, dtrain,
                    early_stopping_rounds=100,
                    evals=[(dvalid,'valid'), (dtrain,'train')],
                    verbose_eval=False)
    predictions = mod.predict(dvalid)
    return 1 - roc_auc_score(y_test, predictions)


# In[48]:


ex_df = diabetes_df[list(set(diabetes_df.columns) & set(lifestyle_cols))]
X_train, X_test, y_train, y_test = train_test_split(ex_df, diabetes_df[['diabetes']], test_size=0.2)
dtrain = xgb.DMatrix(data=X_train.values,
                     feature_names=X_train.columns,
                     label=y_train.values)
dvalid = xgb.DMatrix(data=X_test.values,
                     feature_names=X_test.columns,
                     label=y_test.values)


# In[49]:


def get_params(score):    
    best = fmin(score, space, algo=tpe.suggest, max_evals=250)
    return best


# In[50]:


get_params(xg_score)


# In[51]:


#Note: hpchoice actually returns the index, so the value for max_depth is 3
best_params = {'colsample_bytree': 0.8,
                'eta': 0.25,
                'gamma': 0.95,
                'max_depth': 3,
                'min_child_weight': 3.0,
                'subsample': 0.9,
                'scale_pos_weight': 10,
                'eval_metric': 'auc'}


# In[52]:


fig, ax = plt.subplots(2,2, figsize=(15,15))
for (i, target),df in zip(enumerate(targets), target_dfs):
    
    xgc = XGBClassifier(**best_params)
    y_test, preds = nhanes_pred(xgc, target, df)
    mat = confusion_matrix(y_test, preds)
    sns.heatmap(mat, ax=ax[i%2, i//2], square=True, annot=True, cbar=False, fmt='d').set_title(target)
    print(target, ' F1 score: ', f1_score(y_test,preds))
plt.show()


# In[53]:


xgc = XGBClassifier(**best_params)
y_test, probs = nhanes_pred(xgc, 'diabetes', diabetes_df, proba=1)
plot_roc(y_test, probs)


# In[54]:


ax = plot_importance(xgc, max_num_features=10)
fig = ax.figure
fig.set_size_inches(5, 10)
