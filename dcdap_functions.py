# import statements
# data frame manipulation
import numpy as np
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# statistical analysis
from scipy import stats
# perform multiple pairwise comparison (Tukey HSD)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison


#note that these functions are not generalized and are specific to the DCDAP dataset

def customers_by_method(df):
    # Define colors for each category
    colors = ['blue', 'green', 'gold']

    # Plot the bar chart with the specified colors
    ax = df['sales_method'].value_counts().plot(kind='bar', figsize=(10,5), color=colors)
    plt.title('Sales Method')
    plt.ylabel('Number of Sales')
    plt.xlabel('Sales Method')

    # Add count of each category to the top of the bars
    for i, v in enumerate(df['sales_method'].value_counts()):
        ax.text(i - 0.1, v + 15, str(v), weight='bold')

    plt.show()

def sales_distribution_hist(df):
    ax, fig = plt.subplots(figsize=(15, 5))
    # using the Freedman-Diaconis rule to calculate the optimal number of bins for a histogram
    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    h = 2 * (np.percentile(df['revenue'], 75) - np.percentile(df['revenue'], 25)) * (len(df['revenue']) ** (-1/3))
    bins = int((df['revenue'].max() - df['revenue'].min()) / h)
    # Create a histogram with hue
    sns.histplot(data=df, x='revenue', kde=True, bins=bins, hue='sales_method', palette=['blue', 'green', 'gold'])
    plt.title('Revenue Distribution by Sales Method', fontsize=20)
    plt.xlabel('Revenue', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.show()

def revenue_over_time(df):
    # revenue over time

    # all methods
    df_revenue = df.groupby(['week'])['revenue'].sum().reset_index()

    #split by sales method
    df_rev_email = df[df['sales_method'] == 'email'].groupby(['week'])['revenue'].sum().reset_index()
    df_rev_phone = df[df['sales_method'] == 'call'].groupby(['week'])['revenue'].sum().reset_index()
    df_rev_both = df[df['sales_method'] == 'both'].groupby(['week'])['revenue'].sum().reset_index()


    plt.figure(figsize=(15,5))
    plt.plot(df_revenue['week'], df_revenue['revenue'], color='purple')
    plt.plot(df_rev_email['week'], df_rev_email['revenue'], color='green')
    plt.plot(df_rev_phone['week'], df_rev_phone['revenue'], color='blue')
    plt.plot(df_rev_both['week'], df_rev_both['revenue'], color='gold')
    plt.title('Revenue over time', fontsize=14)
    plt.xlabel('Week', fontsize=14)
    plt.ylabel('Revenue', fontsize=14)
    plt.legend(['All', 'Email', 'Phone', 'Both'])
    plt.show()

def individual_method_rev(df):
    # plot the revenue distribution by sales method
    email_rev = df[df['sales_method'] == 'email']['revenue']
    both_rev = df[df['sales_method'] == 'both']['revenue']
    phone_rev = df[df['sales_method'] == 'call']['revenue']

    # create 3 subplots
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # first histogram
    sns.histplot(email_rev, ax=ax1, color='blue', kde=True)
    ax1.set_title('Email Revenue')
    ax1.set_xlabel('Revenue')
    ax1.set_ylabel('Count')
    ax1.set(xlim=(60, 145))

    # second histogram
    sns.histplot(both_rev, ax=ax2, color='green', kde=True)
    ax2.set_title('Both Revenue')
    ax2.set_xlabel('Revenue')
    ax2.set_ylabel('Count')

    # third histogram
    sns.histplot(phone_rev, ax=ax3, color='red', kde=True)
    ax3.set_title('Phone Revenue')
    ax3.set_xlabel('Revenue')
    ax3.set_ylabel('Count')
    ax3.set(xlim=(20, 80))

    plt.show()

def revenue_method_anova(df):
    revenue = df['revenue']
    sales_method = df['sales_method']

    f_value, p_value = stats.f_oneway(revenue[sales_method == 'email'], 
                                  revenue[sales_method == 'call'], 
                                  revenue[sales_method == 'both'])

    # print the results
    return f_value, p_value

def revenue_method_tukey(df):

    revenue = df['revenue']
    sales_method = df['sales_method']

    # perform the pairwise t-tests with Bonferroni correction
    tukey_results = pairwise_tukeyhsd(revenue, sales_method, 0.05)
    print(tukey_results)
