from flask import Flask, render_template, url_for, request,jsonify
from bs4 import BeautifulSoup
import hashlib
from preprocess import prepare
from preprocess import search as website_search

app= Flask(__name__)

hashes= []


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET'])
def search():
    data=request.args.get("input")
    results=[r[0] for r in website_search(data,2)]
    return jsonify(results=results)


@app.route('/save', methods=['POST','OPTIONS'])
def save():
    if request.method == 'POST':
        data = request.get_json()  
        soup= BeautifulSoup(data["html"],'html.parser')
        hashed_address=hashlib.sha256(
            data["address"].encode('utf16')).hexdigest()

        data["html"]=soup.get_text().replace("\n"," ")
        data["hash_code"]=hashed_address
        
        prepare(data)
        
        # !! This is to modify the test data
        # folder_path= "back_up/wikis"
        # name="0.txt"
        # if len(os.listdir(folder_path))!=0:
        #     name=sorted([int(re.search(r'\d+',filename).group()) for filename in os.listdir(folder_path)])[-1]+1

        # with open(f'{folder_path}/{str(name)}.txt','w',encoding="utf-8",errors="replace") as f:
        #     #f.write(data["address"]+"\n")
        #     f.write(soup.get_text().replace("\n"," ")+"\n")
        #     #f.write("\n")
        
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
        
        return render_template('index.html'),headers
    
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        }
        return '', 200, headers


if __name__=="__main__":
    app.run(debug=True)
