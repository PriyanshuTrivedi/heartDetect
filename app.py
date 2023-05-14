from flask import Flask,render_template,request
import pickle
import numpy as np
model=pickle.load(open("code/test.pkl","rb"))

app = Flask(__name__)

@app.route('/results',methods=["GET","POST"])
def main_form():
    flag=1
    if request.method=="POST":
        patient_name=request.form["patient_name"]
        age=request.form["age"]
        sex=request.form["gender"]
        chol=request.form["chol"]
        trestbps=request.form["trestbps"]
        fbs=request.form["fbs"]
        cp=request.form["cp"]
        restecg=request.form["restecg"]
        exang=request.form["exang"]
        thalach=request.form["thalach"]
        oldpeak=request.form["oldpeak"]
        slope=request.form["slope"]
        ca=request.form["ca"]
        thal=request.form["thal"]
        vals=fvals={}
        vals.update({'age':age,'sex':sex,'chol':chol,'trestbps':trestbps,'fbs':fbs,'cp':cp,'restecg':restecg,'exang':exang,'thalach':thalach,'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal})
        for item in vals.values():
            if item=='':
                flag=0
                break
            print(item)
        if flag:
            age=int(age)
            sex=int(sex)
            chol=int(chol)
            trestbps=int(trestbps)
            fbs=int(fbs)
            cp=int(cp)
            restecg=int(restecg)
            exang=int(exang)
            thalach=int(thalach)
            oldpeak=float(oldpeak)
            slope=int(slope)
            ca=int(ca)
            thal=int(thal)
            fvals.update({'patient_name':patient_name,'age':age,'sex':sex,'chol':chol,'trestbps':trestbps,'fbs':fbs,'cp':cp,'restecg':restecg,'exang':exang,'thalach':thalach,'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal})
            values=[age, sex, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, ca,]
            cp_0=cp_1=cp_2=cp_3=thal_0=thal_1=thal_2=thal_3=slope_0=slope_1=slope_2=0
            if cp==0:
                cp_0=1
            elif cp==1:
                cp_1=1
            elif cp==2:
                cp_2=1
            else:
                cp_3=1
            if thal==0:
                thal_0=1
            elif thal==1:
                thal_1=1
            elif thal==2:
                thal_2=1
            else:
                thal_3=1
            if slope==0:
                slope_0=1
            elif slope==1:
                slope_1=1
            else:
                slope_2=1
            values.extend([cp_0,cp_1,cp_2,cp_3,thal_0,thal_1,thal_2,thal_3,slope_0,slope_1,slope_2])
            print(values)
            output=model.predict([values])
            print(f'priyanshu {output}')
            output=1
    if flag:
        return render_template('results.html',output=output,fvals=fvals)
    return render_template('index.html')

@app.route('/About')
def info_app():
    return render_template('about.html')

@app.route('/aboutDevelopers')
def info_developers():
    return render_template('aboutDevelopers.html')

@app.route('/')
def results():
    return render_template('index.html')
    
if __name__=="__main__":
    app.run(debug=True)