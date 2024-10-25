from flask import Flask, render_template, request
import joblib
import pandas as pd

# Crear la aplicación Flask y especificar la carpeta de plantillas
app = Flask(__name__, template_folder='templates')

# Cargar el modelo entrenado
model = joblib.load('modelo_random_forest.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    edad = request.form['edad']
    sexo = request.form['sexo']
    est_civil = request.form['est_civil']
    hijos = request.form['hijos']
    trabaja_estudia = request.form['trabaja_estudia']
    vive_solo = request.form['vive_solo']
    nivel_socio_econ = request.form['nivel_socio_econ']
    semestre = request.form['semestre']
    pp_anterior = request.form['pp_anterior']

    # Crear un DataFrame con los datos ingresados por el usuario
    new_student = pd.DataFrame({
        'EDAD': [int(edad)],
        'SEXO': [int(sexo)],
        'EST_CIVIL': [int(est_civil)],
        'HIJOS': [int(hijos)],
        'TRABAJA_ESTUDIA': [int(trabaja_estudia)],
        'VIVE_SOLO': [int(vive_solo)],
        'NIVEL_SOCIO_ECON': [int(nivel_socio_econ)],
        'SEMESTRE': [int(semestre)],
        'PPAnterior': [float(pp_anterior)]
    })

    # Hacer la predicción usando el modelo
    predicted_ppacumulado = model.predict(new_student)

    # Devolver el resultado a la página web
    return render_template('index.html', prediction=predicted_ppacumulado[0])

if __name__ == '__main__':  
    app.run(host="0.0.0.0", port=4000, debug=True)
