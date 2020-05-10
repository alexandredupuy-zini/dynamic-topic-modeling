import pandas as pd

def handle_errors(raw_data : pd.DataFrame) : 
    init_n_obs=raw_data.shape[0]
    print('Starting number of verbatim : {}'.format(init_n_obs))

    #Dropping nan
    raw_data.dropna(subset=['raisons_recommandation','Date'],inplace=True)
    no_na_n_obs=raw_data.shape[0]
    print('Number of observations after deleting missing values : {} = {} missing values'.format(no_na_n_obs,init_n_obs-no_na_n_obs))

    #Dropping errors on date 
    no_err=[]
    for i in range(no_na_n_obs) : 
        try : 
            pd.to_datetime(raw_data['Date'].iloc[i])
            no_err.append(i)
        except : 
            pass
    raw_data=raw_data.iloc[no_err].copy()
    final_n_obs=raw_data.shape[0]
    print('Final number of observations after handling errors on date : {} = {} errors on date'.format(final_n_obs,no_na_n_obs-final_n_obs))
    print('Deleted a total of {} observations.'.format(init_n_obs-final_n_obs))


    return raw_data