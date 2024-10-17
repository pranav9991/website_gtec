from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('website.pkl', 'rb'))

# Function to make predictions
def predict_cost(features):
    columns = [
        'Number of Pages',
        'Stock Images_targuided',
        'SEO Design_targuided',
        'Analytics',
        'Image gallery',
        'Live chat',
        'Video gallery',
        'WhatsApp',
        'Appointment scheduling',
        'Chatbot',
        'Login',
        'None',
        'Payment',
        'New or Redesign_targuided',
        'UI/UX Design_targuided',
        'Content Writing Services_targuided',
        'E-commerce Functionality_targuided'
    ]
    df = pd.DataFrame([features], columns=columns)
    prediction = model.predict(df)
    return prediction[0]

# Home page (form to collect input)
@app.route('/')
def index():
    return render_template('index.html')

# Prediction result page
@app.route('/predict', methods=['POST'])
def predict():
    # Extracting form data
    new_or_redesign = request.form['new_or_redesign']
    num_pages = int(request.form['num_pages'])
    ui_ux_design = request.form['ui_ux_design']
    ecommerce_functionality = request.form['ecommerce_functionality']
    basic_features = request.form.getlist('basic_features')
    advanced_features = request.form.getlist('advanced_features')
    content_writing = request.form.get('content_writing')
    stock_images = request.form.get('stock_images')
    seo_design = request.form.get('seo_design')

    # Mapping form data to the feature dictionary
    features = {
        'New or Redesign_targuided': 1 if new_or_redesign == 'New' else 0,
        'Number of Pages': num_pages,
        'UI/UX Design_targuided': {'Standard': 0, 'Advanced': 1, 'Custom': 2}[ui_ux_design],
        'E-commerce Functionality_targuided': 1 if ecommerce_functionality == 'Yes' else 0,
        'Content Writing Services_targuided': 1 if content_writing else 0,
        'SEO Design_targuided': 1 if seo_design else 0,
        'Stock Images_targuided': 1 if stock_images else 0,
        'Analytics': 1 if 'Analytics' in basic_features else 0,
        'Image gallery': 1 if 'Image Gallery' in basic_features else 0,
        'Video gallery': 1 if 'Video Gallery' in basic_features else 0,
        'Live chat': 1 if 'Live Chat' in basic_features else 0,
        'WhatsApp': 1 if 'WhatsApp Integration' in basic_features else 0,
        'Appointment scheduling': 1 if 'Appointment Scheduling' in advanced_features else 0,
        'Chatbot': 1 if 'Chatbot' in advanced_features else 0,
        'Login': 1 if 'Login Systems' in advanced_features else 0,
        'Payment': 1 if 'Payment Gateways' in advanced_features else 0,
        'None': 1 if not advanced_features else 0  # If no advanced features, set 'None' to 1
    }

    # Make the prediction
    estimated_cost = predict_cost(features)
    return render_template('result.html', cost=estimated_cost)

if __name__ == '__main__':
    app.run(debug=True)
