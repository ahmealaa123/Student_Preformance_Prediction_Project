import pandas as pd
import gradio as gr
import joblib


model=joblib.load('KNeighborsClassifier.pkl')



def Students_Performance_Prediction_Model(Q,A,ME,ASS,Ag,FE,CG,T):
    try:
        input_data=pd.DataFrame({
            'Quiz01 [10]':[Q],
            'Assignment01 [8]':[A],
            'Midterm Exam [20]':[ME],
            'Assignment02 [12]':[ASS],
            'Assignment03 [25]':[Ag],
            'Final Exam [35]':[FE],
            'Course Grade':[CG],
            'Total [100]':[T]
        })
        prediction=model.predict(input_data)
        if prediction[0]==0:
            return 'G'
        else:
            return 'W'
    except Exception as e:
        return str(e)
gr.Interface(
    inputs=[
        gr.Number(label='Quiz01 [10]'),
        gr.Number(label='Assignment01 [8]'),
        gr.Number(label='Midterm Exam [20]'),
        gr.Number(label='Assignment02 [12]'),
        gr.Number(label='Assignment03 [25]'),
        gr.Number(label='Final Exam [35]'),
        gr.Number(label='Course Grade'),
        gr.Number(label='Total [100]')
        
    ],
    fn=Students_Performance_Prediction_Model,
    outputs=gr.Textbox(label='Prediction Risk'),
    title='prediction Program',
    description='This program for predict Score Risk of Students'
).launch()
