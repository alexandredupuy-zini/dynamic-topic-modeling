import requests
import pandas as pd 

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None
def save_response_content(response,destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)	

def download_file_from_google_drive(id,destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    print('Downloading dataset.....')
    save_response_content(response, destination)   
    print('Done downloading')

    df=pd.read_csv(destination)
    df.rename({'year':'timestamp'},axis=1,inplace=True)
    return df