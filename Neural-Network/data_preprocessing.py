# DATA PRE-PROCESSING

# import packages
import pandas as pd


# CLEANING THE DATA

def dataset_cleaning(dataset_path):
    # loading the .csv dataset as a dataframe
    df = pd.read_csv(dataset_path)

    # imputing missing 'Age' column values with the mean of the 'Age' column
    age_mean=df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)


    # imputing missing 'Weight' column values
    # find mean of the 'Weight' column, without considering the outliers in the mean calculation
    #impute the missing values with this mean
    Q1 = df['Weight'].quantile(0.25)
    Q3 = df['Weight'].quantile(0.75)
    IQR = Q3 - Q1

    weight_total=0
    n=0

    for index in df.index:
        if(pd.isna(df['Weight'][index])== False):
            wt=df['Weight'][index]
            
            lower_bound= Q1-1.5*IQR
            upper_bound= Q3 + 1.5*IQR
            
            if(lower_bound <= wt <= upper_bound):
                weight_total+=wt
                n+=1
                
    weight_mean=weight_total/n

    df['Weight'] = df['Weight'].fillna(weight_mean)


    # imputing missing 'Delivery phase' column values with '1'
    df['Delivery phase'] = df['Delivery phase'].fillna(1)


    # imputing missing 'HB' column values with the mean of the 'HB' column
    HB_mean=df['HB'].mean()
    df['HB'] = df['HB'].fillna(HB_mean)


    # imputing missing 'BP' column values
    # find mean of the 'BP' column, without considering the outliers in the mean calculation
    #impute the missing values with this mean

    Q1 = df['BP'].quantile(0.25)
    Q3 = df['BP'].quantile(0.75)
    IQR = Q3 - Q1

    BP_total=0
    n=0

    for index in df.index:
        if(pd.isna(df['BP'][index])== False):
            bp_val=df['BP'][index]
            
            lower_bound= Q1-1.5*IQR
            upper_bound= Q3 + 1.5*IQR
            
            if(lower_bound <= bp_val <= upper_bound):
                BP_total+=bp_val
                n+=1
                
    BP_mean=BP_total/n

    df['BP'] = df['BP'].fillna(BP_mean)



    # imputing all missing 'Education' column values with '5'
    df['Education'] = df['Education'].fillna(5)


    # imputing all missing 'Residence' column values with '1'
    # if the 'Residence' cell value = 2, change it to 1

    df['Residence'] = df['Residence'].fillna(1)


    # one-hot encode the 'Community' column
    df = pd.concat([df,pd.get_dummies(df['Community'], prefix='Community')],axis=1)
    df = df.drop(columns=['Community'])


    for index in df.index:
        # if the value of 'Residence' cell is '2', replace it with '0'
        if(df['Residence'][index]==2):
            df.at[index,'Residence']=0

        # if the value of 'Delivery phase' cell is '2', replace it with 0    
        if(df['Delivery phase'][index]==2):
            df.at[index,'Delivery phase']=0
    
    return df



# function to normalize and standardize the columns of the dataframe
def transform_data(df):
    result = df.copy()    # 'result' will be the returned dataframe
    
    normalize_columns=['Age','Weight','HB','BP']
    
    # Standardizing the 'Age', 'Weight, 'HB', 'BP' columns
    # to have mean = 0 and stadard deviation = 1
    for feature_name in normalize_columns:
            mean_val = df[feature_name].mean()
            std_val = df[feature_name].std()
            result[feature_name] = (df[feature_name] - mean_val) / std_val
   
            
    # for Education feature
    # perform min-max normalization on the 'Education' column
    min_val = 0
    max_val=10
    result['Education'] = (df['Education'] - min_val) / (max_val-min_val)
    
 
    return result



# main function


if __name__=='__main__':
    # clean the dataset and store as dataframe
    df = dataset_cleaning(r'LBW_Dataset_uncleaned.csv')

    # transform the column data
    df = transform_data(df)

    # converting cleaned dataframe into .csv file

    df.to_csv(r'LBW_Dataset_cleaned.csv',index=False)
