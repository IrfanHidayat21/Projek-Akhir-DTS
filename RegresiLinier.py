# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:12:35 2019

@author: ACER-V
"""

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div(children=[
    html.Div(children='''
        
    '''),
    dcc.Input(id='input', value='', type='text'),
    html.H1(children='FINAL PROJECT DTS'),
    html.Div(id='output-graph'),
    
])

@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='input', component_property='value')]
)
 
def update_value(input_data):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as stats

    # Mengimpor dataset
    dataset = pd.read_csv('produksi.csv')
    X = dataset.iloc[:, :-1].values
    Tampilkan_X = pd.DataFrame(X) #visualisasi X
    y = dataset.iloc[:, 4].values
  
    dataset.head()

    dataset[dataset['jenis_komoditi']=='Padi']

    #In[5]
    x=dataset[dataset['jenis_komoditi']=='Padi']['luas_panen']
    y=dataset[dataset['jenis_komoditi']=='Padi']['jumlah_produksi']
    plt.plot(x,y, 'r')
    plt.xlabel('Tahun')
    plt.ylabel('Produksi')
    plt.title('Jumlah Produksi Panen Pertahun\n')

    #In[14]
    kemiringan, konstanta, r_value, p_value, std_err = stats.linregress (x,y)
    garis_regresi=kemiringan*x+konstanta
    
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr
    korelasi_pearson, pvalue=pearsonr (x,y)
    korelasi_spearman, pvalue=spearmanr (x,y)
    print('Korelasi Pearson = %.2f; Korelasi Spearman = %.2f' % (korelasi_pearson,korelasi_spearman))
    
    plt.scatter( x,y, color ='green')
    plt.plot(x,garis_regresi,'r')
    plt.xlabel('Tahun')
    plt.ylabel('Produksi/Ton')
    plt.title('Jumlah Produksi Padi Pertahun\n')

    #In[15]
    np.polyfit (x, y, 2)

    #In[21]
    theta0=np.polyfit (x,y,2) [0]
    theta1=np.polyfit (x,y,2) [1]
    theta2=np.polyfit (x,y,2) [2]

    #ytopi2=theta0+theta1+theta2*x**2
    ytopi2=theta0*x**2+theta1*x+theta2
    #cx^2+bx+a

    plt.scatter( x,y, color ='green')
    plt.plot(x, ytopi2,'black')
    plt.xlabel('Tahun')
    plt.ylabel('Produksi/Ton')
    plt.title('Jumlah Produksi Padi Pertahun\n')

    #In[28]
    theta0=np.polyfit (x,y,4) [0]
    theta1=np.polyfit (x,y,4) [1]
    theta2=np.polyfit (x,y,4) [2]
    theta3=np.polyfit (x,y,4) [3]
    theta4=np.polyfit (x,y,4) [4]


    ytopi4=theta0*x**4+theta1*x**3+theta2*x**2+theta3*x+theta4
    #cx^2+bx+a

    plt.scatter( x,y, color ='green', label='Data')
    plt.plot(x,garis_regresi,'red', label='linier')
    plt.plot(x, ytopi2,'r', label='Derajat 2')
    plt.plot(x, ytopi4,'b', label='Derajat 4')
    plt.xlabel('Tahun')
    plt.ylabel('Produksi/Ton')
    plt.legend(loc='best')
    plt.title('Jumlah Produksi Padi Pertahun\n')


    #In[30]
    rmse_linier = np.sqrt(sum((garis_regresi-y)**2)/len(y))
    rmse_derajat2 = np.sqrt(sum((ytopi2-y)**2)/len(y))
    rmse_derajat4 = np.sqrt(sum((ytopi4-y)**2)/len(y))

    print('RMSE untuk regresi linier           = %20.2f' %(rmse_linier))
    print('RMSE untuk polinominal berderajat 2 = %20.2f' %(rmse_derajat2))
    print('RMSE untuk polinominal berderajat 4 = %20.2f' %(rmse_derajat4))

    #In[31]
    ybar = sum(y)/len(y)

    ssregresi_linier = sum((garis_regresi-ybar)**2)
    ssregresi_derajat2 = sum((ytopi2-ybar)**2)
    ssregresi_derajat4 = sum((ytopi4-ybar)**2)

    sstotal = sum((y-ybar)**2)
    
    r2_linier = ssregresi_linier/sstotal
    r2_derajat2 = ssregresi_derajat2/sstotal
    r2_derajat4 = ssregresi_derajat4/sstotal

    print('R^2 untuk regresi linier           = %.2f' %(r2_linier))
    print('R^2 untuk polinominal berderajat 2 = %.2f' %(r2_derajat2))
    print('R^2 untuk polinominal berderajat 4 = %.2f' %(r2_derajat4))

    #start = datetime.datetime(2015, 1, 1)
    #end = datetime.datetime.now()
    #df = web.DataReader(input_data, 'morningstar', start, end)
    #dataset.reset_index(inplace=True)
    #df.set_index("Date", inplace=True)
    #df = df.drop("Symbol", axis=1)

    return dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': x, 'y': y, 'type': 'bar', 'name': 'Data Jumlah Produksi Padi di DKI Jakarta'},
                {'x': x, 'y': garis_regresi, 'type': 'plot', 'name': 'Garis Regresi Data Jumlah Produksi Padi di DKI Jakarta'},
                {'x': x, 'y': ytopi2, 'type': 'plot', 'name': 'Garis Regresi Derajat 2'},
                {'x': x, 'y': ytopi4, 'type': 'plot', 'name': 'Garis Regresi Derajat 4'},
            ],
            'layout': {
                'title': 'Data Jumlah Produksi Padi di DKI Jakarta',
                
            }
        }
    )
    
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)