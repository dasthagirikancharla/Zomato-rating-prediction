import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('zomato.pkl', 'rb'))
k='listed_in(type)'
z='online Order'

m='listed_in(city)'
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        z = int(request.form['online Order'])
        book_table = int(request.form['book_table'])
        votes = int(request.form['votes'])
        location = int(request.form['location'])
        rest_type = int(request.form['rest_type'])
        dish_liked = int(request.form['dish_liked'])
        cuisines = int(request.form['cuisines'])
        approx_cost = int(request.form['approx_cost(for two people)'])
        k = int(request.form['listed_in(type)'])
        m = int(request.form['listed_in(city)'])

    # features = [int(x) for x in request.form.values()]
    # final_features = [np.array(features)]
    # prediction = model.predict(final_features)

    

    # return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))
    data = np.array([[z,book_table,votes,location,rest_type,dish_liked,cuisines,approx_cost,k,m]])
    my_prediction = model.predict(data)
    output = round(my_prediction[0], 1)    
    return render_template('index.html',  prediction_text='Your Rating is: {}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)