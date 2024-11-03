
from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from wildblueberry.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")
    
@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 
 										

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # clonesize,honeybee,bumbles,andrena,osmia,MaxOfUpperTRange,MinOfUpperTRange,AverageOfUpperTRange,
            # MaxOfLowerTRange,MinOfLowerTRange,AverageOfLowerTRange,RainingDays,AverageRainingDays,fruitset,
            # fruitmass,seeds
            #  reading the inputs given by the user
            clonesize =float(request.form['clonesize'])
            honeybee =float(request.form['honeybee'])
            bumbles =float(request.form['bumbles'])
            andrena =float(request.form['andrena'])
            osmia =float(request.form['osmia'])
            MaxOfUpperTRange =float(request.form['MaxOfUpperTRange'])
            MinOfUpperTRange =float(request.form['MinOfUpperTRange'])
            AverageOfUpperTRange =float(request.form['AverageOfUpperTRange'])
            MaxOfLowerTRange =float(request.form['MaxOfLowerTRange'])
            MinOfLowerTRange =float(request.form['MinOfLowerTRange'])
            AverageOfLowerTRange =float(request.form['AverageOfLowerTRange'])
            RainingDays =float(request.form['RainingDays'])
            AverageRainingDays =float(request.form['AverageRainingDays'])
            fruitset =float(request.form['fruitset'])
            fruitmass =float(request.form['fruitmass'])
            seeds =float(request.form['seeds'])
           
       
         
            data = [clonesize,honeybee,bumbles,andrena,osmia,MaxOfUpperTRange,MinOfUpperTRange,AverageOfUpperTRange,
                    MaxOfLowerTRange,MinOfLowerTRange,AverageOfLowerTRange,RainingDays,AverageRainingDays,fruitset,
                    fruitmass,seeds]
            data = np.array(data).reshape(1, 16)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('result.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	# app.run(host="0.0.0.0", port = 8080) #for AWS /Local host
    app.run(host="0.0.0.0", port = 80) # Asure 

from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from wildblueberry.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")
    
@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 
 										

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # clonesize,honeybee,bumbles,andrena,osmia,MaxOfUpperTRange,MinOfUpperTRange,AverageOfUpperTRange,
            # MaxOfLowerTRange,MinOfLowerTRange,AverageOfLowerTRange,RainingDays,AverageRainingDays,fruitset,
            # fruitmass,seeds
            #  reading the inputs given by the user
            clonesize =float(request.form['clonesize'])
            honeybee =float(request.form['honeybee'])
            bumbles =float(request.form['bumbles'])
            andrena =float(request.form['andrena'])
            osmia =float(request.form['osmia'])
            MaxOfUpperTRange =float(request.form['MaxOfUpperTRange'])
            MinOfUpperTRange =float(request.form['MinOfUpperTRange'])
            AverageOfUpperTRange =float(request.form['AverageOfUpperTRange'])
            MaxOfLowerTRange =float(request.form['MaxOfLowerTRange'])
            MinOfLowerTRange =float(request.form['MinOfLowerTRange'])
            AverageOfLowerTRange =float(request.form['AverageOfLowerTRange'])
            RainingDays =float(request.form['RainingDays'])
            AverageRainingDays =float(request.form['AverageRainingDays'])
            fruitset =float(request.form['fruitset'])
            fruitmass =float(request.form['fruitmass'])
            seeds =float(request.form['seeds'])
           
       
         
            data = [clonesize,honeybee,bumbles,andrena,osmia,MaxOfUpperTRange,MinOfUpperTRange,AverageOfUpperTRange,
                    MaxOfLowerTRange,MinOfLowerTRange,AverageOfLowerTRange,RainingDays,AverageRainingDays,fruitset,
                    fruitmass,seeds]
            data = np.array(data).reshape(1, 16)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('result.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	# app.run(host="0.0.0.0", port = 8080) #for AWS /Local host
    app.run(host="0.0.0.0", port = 80) # Asure 