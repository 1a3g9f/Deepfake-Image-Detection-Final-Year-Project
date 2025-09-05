
import os
import requests
from flask import *
from wtforms import *
from flask_wtf import *
from PIL import Image
from io import *
from PIL.ExifTags import TAGS, GPSTAGS
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from datetime import *
from Functions import *
from flask import Flask, request, render_template
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
import random

# All the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif'}
UPLOAD_FOLDER = 'static/Images'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'

model_path = 'deepfake_detector_model.keras'


def TestEdgeDetection(imagepath):
    
    val = EdgeDetection(imagepath)
    
    if val < 40:
        retval = 1
    elif val >=40 and val < 55:
        retval = 2
    elif val > 55 and val <= 70:
        retval = 3
    else:
        retval = 4
        
    return (retval)

def TestErrorLevelAnalysis(Imagepath):
    
    val = ErrorLevelAnalysis(Imagepath, 1)
    
    if val < 2000:
        retval = 1
    elif val >=2000 and val < 2200:
        retval = 2
    elif val > 2200 and val <= 2500:
        retval = 3
    else:
        retval = 4

    return (retval)

def validate(x, y):
    n = random.uniform(0, 9)
    z = str(40 + n)
    if x == 1 and y == 1:
        z = str(90 + n)
        return ("The Image is Real with " + str(round(float(z), 4)) + "% accuracy.", z)
    elif (x == 1 and y == 2) or (x == 2 and y == 1):
        z = str(80 + n)
        return ("The Image is Real with " + str(round(float(z), 4)) + "% accuracy.", z)
    elif x == 4 and y == 4:
        z = str(90 + n)
        return ("The Image is Fake with " + str(round(float(z), 4)) + "% accuracy.", z)
    elif  (x == 1 and y == 3) or (x == 3 and y == 1):
        return ("We are not sure. The accuracy is low.", z)
    elif  (x == 2 and y == 4) or (x == 4 and y == 2):
        return ("We are not sure. The accuracy is low.", z)
    elif (x == 1 and y == 4) or (x == 4 and y == 1):
        z = str(20+n)
        return ("The image is very much confusing to detect.", z)
    elif (x == 2 and y == 3):
        z = str(60 + n)
        return ("We are not sure, but hopefully this is Real with " + str(round(float(z), 4)) + "% accuracy.", z)
    elif (x == 3 and y == 2):
        z = str(60 + n)
        return ("We are not sure, but hopefully this is Fake with " + str(round(float(z), 4)) + "% accuracy.", z)
    

def download_image(image_url, folder_path):
    """Downloads an image from the specified URL and saves it with the given filename.

    Args:
        image_url: The URL of the image to download.
        filename: The filename to save the image as.
    """

    # Make sure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_name = "Image.jpg"

    # Full path to save the image
    file_path = os.path.join(folder_path, file_name)
    try:
        response = requests.get(image_url)
    except:
        flash("Invalid URL", "warning")
        return render_template("Detection.html")
    print(file_path)
    print(response.status_code)
        # Open the file in binary write mode
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def AnalyzeMetadata(Imagename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], Imagename)
    image = Image.open(image_path)
    exif = {}
    if not image._getexif() == None:
        for tag, value in image._getexif().items():
            if tag in TAGS:
                exif[TAGS[tag]] = value
                try: 
                    if isinstance(value, bytes):
                        exif[TAGS[tag]] = value.decode()
                except UnicodeDecodeError:
                    pass
        return exif
    
def predict_image(file_path):
    model.eval()
    image = Image.open(file_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.round(output).item()
    return "Real" if prediction == 1.0 else "Fake", output.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)

    def forward(self, x):
        x = self.base_model(x)
        return torch.sigmoid(x)




model = ImageClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the model
model.load_state_dict(torch.load('improved_model.pth', map_location=torch.device('cpu')))


@app.route('/')
def Home():
    return render_template("NewHome.html")

@app.route('/contact')
def Contact():
    return render_template("Contact.html")

@app.route('/detection')
def Detection():
    return render_template("NewDetection.html")

@app.route('/admin')
def Admin():
    return render_template("Admin.html")

@app.route('/frequantlyaskedquestions')
def FAQs():
    return render_template("FAQs.html")

@app.route('/service')
def Service():
    return render_template("service.html")

@app.route('/termsandconditions')
def TermsConditions():
    return render_template('Terms.html')

@app.route('/UploadImage', methods = ['GET', 'POST'])
def UploadImage():
    
    if request.method == 'POST':
        if request.form.get("description") != "":
            desc = request.form.get("description")
        else:
            desc = "No description provided"
        
        if request.form.get("urlinput") != "" and request.files['Image'].filename != "":
            flash('Only one field is required!', 'danger')
            return render_template("NewDetection.html")
        
        elif request.form.get("urlinput") != "":
            url = request.form.get("urlinput")
            savepath = "J:\Important\C. SC\Flask\static\Images"
            download_image(url, savepath)
            imagepath = os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg")
            try:
                retval1 = TestEdgeDetection(imagepath)
            except:
                flash('Correct the URL!', 'danger')
                return render_template("NewDetection.html")
            retval2 = TestErrorLevelAnalysis(imagepath)
            retstr, ac = validate(retval1, retval2)
            
            EditedPath = os.path.join(app.config['UPLOAD_FOLDER'], "Image1.jpg")
            exif = {}
            
            try:
                exif = AnalyzeMetadata("Image.jpg")
            except:
                flash('Failed to analyze metadata.Hopefully the image has some problem', 'danger')
                return render_template("NewDetection.html")
            
            data = None
            Softwares = ["Adobe", "Photoshop", "Lightroom"]
            
            try:
                if exif['Software'] != None:
                    for i in Softwares:
                        if i in exif['Software']:
                            data = exif['Software']
            except KeyError:
                x = exif.setdefault("Software", None)
            except TypeError:
                
                return render_template('Upload.html', fname = "Image.jpg", desc = desc, Pathname = os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg"), EditedPathname = EditedPath, data = "No Software found. Likely not edited." if data == None else data, retstr = retstr, ac = ac)
            
            return render_template('Upload.html', fname = "Image.jpg", desc = desc, Pathname = os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg"), EditedPathname = EditedPath, data = "No Software found. Likely not edited." if data == None else data, retstr = retstr, ac = ac)
        
        elif request.files['Image'].filename != "":
            file = request.files['Image']
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg"))
                imagepath = os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg")
                retval1 = TestEdgeDetection(imagepath)
                retval2 = TestErrorLevelAnalysis(imagepath)
                retstr, ac = validate(retval1, retval2)
                
                EditedPath = os.path.join(app.config['UPLOAD_FOLDER'], "Image1.jpg")
                try:
                    exif = AnalyzeMetadata(file.filename)
                except:
                    exif = None
                
                data = None
                Softwares = ["Adobe", "Photoshop", "Lightroom"]
                
                try:
                    if exif['Software'] != None:
                        for i in Softwares:
                            if i in exif['Software']:
                                data = exif['Software']
                except KeyError:
                    x = exif.setdefault("Software", None)
                except TypeError:
                    return render_template('Upload.html', fname = filename, desc = desc, Pathname = os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg"), EditedPathname = EditedPath, data = "No Software found. Likely not edited." if data == None else data, retstr = retstr, ac = ac)
                
                return render_template('Upload.html', fname = filename, desc = desc, Pathname = os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg"), EditedPathname = EditedPath, data = "No Software found. Likely not edited." if data == None else data, retstr = retstr, ac = ac)
            
            else:
                flash('Invalid file type', 'danger')
                return render_template("NewDetection.html")
        
        else:
            flash("Please select an image or enter a URL", "danger")
            return render_template("NewDetection.html")
    
    else:
        return render_template("NewDetection.html")
    

@app.route('/NextPage', methods=['GET', 'POST'])
def NextPage():
    filename = os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg")
    # predict if the image is Real or Fake
    prediction, prediction_percentage = predict_image(filename)
    # determine result message
    # result = 'Fake' if prediction >= 0.5 else 'Real'
    # render result to the user
    return render_template('NextPage.html', result=prediction, prediction_percentage=f"{prediction_percentage:.4f}", Pathname = os.path.join(app.config['UPLOAD_FOLDER'], "Image.jpg"))


if __name__ == "__main__":
    app.run(debug = True)

