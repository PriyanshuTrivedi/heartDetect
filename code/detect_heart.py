import pickle
import numpy as np


model=pickle.load(open("code/test.pkl","rb"))

def predict(values):
    output=model.predict([values])
    return output

if __name__=="__main__":
        values={}
        columns=['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'ca']

        for item in columns:
            values[item]=float(input(f"Enter Your {item} : "))
        
        values['cp_0']=values['cp_1']=values['cp_2']=values['cp_3']=values['thal_0']=values['thal_1']=values['thal_2']=values['thal_3']=values['slope_0']=values['slope_1']=values['slope_2']=0

        cp=input("Enter Your cp : ")
        thal=input("Enter Your thal : ")
        slope=input("Enter Your slope : ")

        values[f'cp_{cp}']=values[f'cp_{thal}']=values[f'cp_{slope}']=1
        
        output=model.predict_proba([np.array(list(values.values()))])
        
        print(output)

        