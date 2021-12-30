

# =============================================================================
# Assignment Proxima
# =============================================================================


# Load all needed modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import datasets
df_0 = pd.read_excel('https://drive.google.com/uc?export=download&id=1zcAObv2swOcR5vZD_DTVGgC78hqf2GnZ', sheet_name=0, header=0)

df_1 = pd.read_csv('https://drive.google.com/uc?export=download&id=1n_CVbn4GluIvXn86EutbXwQR-pNabS-a', header=0, sep=';', encoding='cp1251')

df_2 = pd.read_excel('https://drive.google.com/uc?export=download&id=19WgS_gBR5zvGkLnC5LvBvNwPcRFilp-j', sheet_name=0, header=0, skiprows=3, usecols=lambda x: 'Unnamed' not in x)

df_id = pd.read_excel('https://drive.google.com/uc?export=download&id=1RpO_9mxyxoA5ATK6ob5UdeCCkaSUalJR', sheet_name=0, header=0)

df_period = pd.read_excel('https://drive.google.com/uc?export=download&id=1adj2j5k6Q6_O02epS-CQvwSasCHjNf-t', sheet_name=0, header=0)


# Check data types, indexes, NaN values
df_0.info()
df_0.columns
df_0.head(10)

df_1.info()
df_1.columns
df_1.head(10)

df_2.info()
df_2.columns
df_2.head(10)

df_id.info()
df_id.columns
df_id.head(10)

df_period.info()
df_period.columns
df_period.head(10)



# Convert df1 to a plain table
# Add empty PERIOD_ID
for i in range(len(df_1['DRUGS_ID'])):
    if str(df_1['DRUGS_ID'][i]) == 'nan':
        df_1.loc[i,'DRUGS_ID'] = df_1['DRUGS_ID'][i-1]
        
# Melt datetime columns to a rows
df_1_melt = df_1.melt(id_vars=['DRUGS_ID', 'Тип'], var_name='PERIOD_ID')
df_1_melt

# Convert data types
df_1_melt[['value']] = df_1_melt['value'].str.replace(",", ".").astype("float")
df_1_melt[['DRUGS_ID']] = df_1_melt[['DRUGS_ID']].astype("int64")
df_1_melt[['PERIOD_ID']] = df_1_melt[['PERIOD_ID']].astype("int64")


# Make pivot table from 'type' column 
df_1_pivot = pd.pivot_table(df_1_melt, 
                  index = ['PERIOD_ID', 'DRUGS_ID'], 
                  columns = 'Тип', 
                  values = 'value').reset_index().rename_axis(None, axis=1)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_1 = df_1_pivot[['PERIOD_ID', 'DRUGS_ID', 'VOLUME', 'QUANTITY']]


# change data type for column in df2
df_2['QUANTITY'] = df_2['QUANTITY'].apply(pd.to_numeric, errors='coerce').fillna(0)
df_2.info()


## Quick statistics
df_0.describe()
df_1.describe()
df_2.describe()
df_period.describe()
df_id.describe()




## Join tables

# We can connect all datasets in SQL style
# Let's join ver_o, Drugs_id, PERIOD_ID
drugs_connect = pd.merge(df_0, df_id, on='DRUGS_ID', how="left")
all_connect = pd.merge(drugs_connect, df_period, on='PERIOD_ID', how="left")
all_connect.info()

drugs_sales = all_connect[['Date', 'Brand', 'Market Org', 'Full medication name', 'VOLUME', 'QUANTITY']]
drugs_sales.info()
drugs_sales




## Group analysis

# count number of unique entries
drugs_sales.nunique()


## group with one variable

# best selling Brands
drugs_brand = drugs_sales.groupby(['Brand'], as_index=False).sum().sort_values(['QUANTITY'], ascending=False)
drugs_brand

# Visualize brand sales using barplot
sns.set_style('darkgrid')
sns.barplot(x='Brand', y='QUANTITY', data=drugs_brand, ci = None)
plt.xticks(rotation = 'vertical')
plt.ticklabel_format(style='plain', axis='y')


# best selling Market Org
drugs_org = drugs_sales.groupby(['Market Org'], as_index=False).sum().sort_values(['QUANTITY'], ascending=False)
drugs_org

# Visualize sales by organization using barplot
sns.set_style('darkgrid')
sns.barplot(x='Market Org', y='QUANTITY', data=drugs_org, ci = None)
plt.xticks(rotation = 'vertical')
plt.ticklabel_format(style='plain', axis='y')


# sort sales by date
drugs_date = drugs_sales.groupby(['Date'], as_index=False).sum().sort_values(['QUANTITY'], ascending=False)
drugs_date
drugs_date[['Date']] = drugs_date['Date'].dt.date


# Visualize sales by date using barplot
sns.set_style('darkgrid')
sns.barplot(x='Date', y='QUANTITY', data=drugs_date, ci = None)
plt.xticks(rotation = 'vertical')
plt.ticklabel_format(style='plain', axis='y')


# sort sales by name
drugs_name = drugs_sales.groupby(['Full medication name'], as_index=False).sum().sort_values(['QUANTITY'], ascending=False)
drugs_name



## group with multiple variables

# Grouping by Date and Brand, sort by QUANTITY
drugs_d_b = drugs_sales.groupby(['Date', 'Brand']).sum().groupby(level=0, group_keys=False).apply(lambda x: x.sort_values('QUANTITY', ascending=False))
drugs_d_b


# Grouping by Date and Brand, sort by VOLUME
drugs_sales.groupby(['Date', 'Brand']).sum().groupby(level=0, group_keys=False).apply(lambda x: x.sort_values('VOLUME', ascending=False))


# Visualize grouped table using Scatterplot with varying point sizes
sns.set_theme(style="white")
sns.relplot(x="Date", y="Brand", size="QUANTITY",\
            sizes=(40, 400), palette="muted",\
            height=6, data=drugs_d_b)
plt.xticks(rotation = 'vertical')


sns.set_theme(style="white")
sns.relplot(x="Date", y="Brand", size="VOLUME",\
            sizes=(40, 400), palette="muted",\
            height=6, data=drugs_d_b)
plt.xticks(rotation = 'vertical')



# Grouping with a pivot method
grouped = all_connect[['Date', 'Brand', 'QUANTITY']].groupby(['Date', 'Brand'],as_index=False).sum()
grouped_pivot = grouped.pivot(index='Date',columns='Brand')
grouped_pivot = grouped_pivot.fillna(0)
grouped_pivot.index = grouped_pivot.index.strftime('%d-%m-%Y')
grouped_pivot

# Visualize pivot table using Heatmap 
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
plt.xticks(rotation=90)
fig.colorbar(im)
plt.show()



## growth rate

# find out growth rate in percents by period
drugs_date_grow = drugs_sales.groupby(['Date'], as_index=False).sum().sort_values(['Date'])
drugs_date_grow

drugs_date_grow['v_chg']=drugs_date_grow['VOLUME'].pct_change()
drugs_date_grow['q_chg']=drugs_date_grow['QUANTITY'].pct_change()
drugs_date_grow


# find out growth rate in percents for brand by period

# prepare grouping
drugs_d_b_grow = drugs_sales.groupby(['Date', 'Brand'], as_index=False).sum().sort_values(['Brand', 'Date'])
drugs_d_b_grow['q_chg'] = drugs_d_b_grow.groupby(['Brand','Date'])['QUANTITY'].pct_change()
drugs_d_b_grow

# calculate growth rate for brand by period in corresponding columns
drugs_d_b_grow['v_chg'] = drugs_d_b_grow.sort_values('Date').groupby('Brand').VOLUME.pct_change()

drugs_d_b_grow['q_chg'] = drugs_d_b_grow.sort_values('Date').groupby('Brand').QUANTITY.pct_change()

drugs_d_b_grow = drugs_d_b_grow.fillna(0)

drugs_d_b_grow


# Visualize growth rate for brand by period

# datasets with growth rate
d_q = drugs_d_b_grow[['Date', 'Brand', 'q_chg']]
d_v = drugs_d_b_grow[['Date', 'Brand', 'v_chg']]

# prepare pivot tables for plotting
data_q = d_q.pivot(index='Date',columns='Brand', values='q_chg').rename_axis(None, axis=1).fillna(0)
data_v = d_v.pivot(index='Date',columns='Brand', values='v_chg').rename_axis(None, axis=1).fillna(0)

data_q
data_v


# Visualize growth rate(QUANTITY) in percents for brand by period using lineplot
sns.set_theme(style="whitegrid")
ax = sns.lineplot(data=data_q, palette="tab10")
plt.legend(fontsize='xx-small')
plt.xticks(rotation = 'vertical')


# Visualize growth rate(VOLUME) in percents for brand by period using lineplot
sns.set_theme(style="whitegrid")
ax = sns.lineplot(data=data_v, palette="tab10")
plt.legend(fontsize='xx-small')
plt.xticks(rotation = 'vertical')






# *

# rename columns
df_0 = df_0.rename(columns={'PERIOD_ID': '0_PERIOD_ID', 'DRUGS_ID': '0_DRUGS_ID', 'VOLUME': '0_VOLUME', 'QUANTITY': '0_QUANTITY'})

df_1 = df_1.rename(columns={'PERIOD_ID': '1_PERIOD_ID', 'DRUGS_ID': '1_DRUGS_ID', 'VOLUME': '1_VOLUME', 'QUANTITY': '1_QUANTITY'})

df_2 = df_2.rename(columns={'PERIOD_ID': '2_PERIOD_ID', 'DRUGS_ID': '2_DRUGS_ID', 'VOLUME': '2_VOLUME', 'QUANTITY': '2_QUANTITY'})

# join tables by version
drugs_connect_0 = pd.merge(df_0, df_id, left_on='0_DRUGS_ID', right_on='DRUGS_ID', how="left")

all_connect_0 = pd.merge(drugs_connect_0, df_period, left_on='0_PERIOD_ID', right_on='PERIOD_ID', how="left")
all_connect_0.info()


drugs_connect_1 = pd.merge(df_1, df_id, left_on='1_DRUGS_ID', right_on='DRUGS_ID', how="left")

all_connect_1 = pd.merge(drugs_connect_1, df_period, left_on='1_PERIOD_ID', right_on='PERIOD_ID', how="left")
all_connect_1.info()


drugs_connect_2 = pd.merge(df_2, df_id, left_on='2_DRUGS_ID', right_on='DRUGS_ID', how="left")

all_connect_2 = pd.merge(drugs_connect_2, df_period, left_on='2_PERIOD_ID', right_on='PERIOD_ID', how="left")
all_connect_2.info()



# Group all sets by Full medication name
drugs_sales_0 = all_connect_0[['Date', 'Brand', 'Market Org', 'Full medication name', '0_VOLUME', '0_QUANTITY']]

drugs_name_0 = drugs_sales_0.groupby(['Full medication name'], as_index=False).sum().sort_values(['0_QUANTITY'], ascending=False)


drugs_sales_1 = all_connect_1[['Date', 'Brand', 'Market Org', 'Full medication name', '1_VOLUME', '1_QUANTITY']]

drugs_name_1 = drugs_sales_1.groupby(['Full medication name'], as_index=False).sum().sort_values(['1_QUANTITY'], ascending=False)


drugs_sales_2 = all_connect_2[['Date', 'Brand', 'Market Org', 'Full medication name', '2_VOLUME', '2_QUANTITY']]

drugs_name_2 = drugs_sales_2.groupby(['Full medication name'], as_index=False).sum().sort_values(['2_QUANTITY'], ascending=False)



# join grouped tables in one
drugs_name_m1 = pd.merge(drugs_name_0, drugs_name_1, on='Full medication name', how='left')

drugs_name_m2 = pd.merge(drugs_name_m1, drugs_name_2, on='Full medication name', how="left")


# 
drugs_name_m2['0-1'] = abs(drugs_name_m2['0_QUANTITY'] - drugs_name_m2['1_QUANTITY'])

drugs_name_m2['0-2'] = abs(drugs_name_m2['0_QUANTITY'] - drugs_name_m2['2_QUANTITY'])


drugs_name_m2[['0-1', '0-2']]
drugs_name_m2[['0-1']].sum()
drugs_name_m2[['0-2']].sum()

# df_2 is less different from production dataset df_0



