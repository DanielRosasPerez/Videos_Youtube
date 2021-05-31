# IMPORTAMOS LAS LIBRERÍAS:
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# DECLARAMOS LOS PARÁMETROS E HIPERPARÁMETROS:
entradas = np.asarray([[-1,1]])
salida_real = np.asarray([[0]])

neuronas_de_entrada = entradas.shape[1]  # entradas.shape[1] = 2
neuronas_de_la_capa_oculta = 3
neuronas_capa_de_salida = 1

pesos_capa_entrada_y_oculta = np.random.random(size=(neuronas_de_entrada,neuronas_de_la_capa_oculta))
pesos_capa_oculta_y_capa_de_salida = np.random.random(size=(neuronas_de_la_capa_oculta, neuronas_capa_de_salida))

biases_capa_oculta = np.zeros(shape=(1,neuronas_de_la_capa_oculta))
biases_capa_salida = np.zeros(shape=(1,neuronas_capa_de_salida))

Matrices_Pesos = [pesos_capa_entrada_y_oculta, pesos_capa_oculta_y_capa_de_salida]
Matrices_Biases = [biases_capa_oculta, biases_capa_salida]

# DECLARAMOS LAS FUNCIONES NECESARIAS PARA ENTRENAR NUESTRA RED NEURONAL:

def sigmoide(suma_ponderada):
    """Está función aplica la función de activación sigmoide a nuestra suma ponderada."""
    activacion = 1/(1+np.exp(-suma_ponderada))
    return activacion

def derivada_sigmoide(valor_activacion):
    return valor_activacion * (1 - valor_activacion)

def error_cuadratico_medio(salida_real, prediccion_red):   
    return (prediccion_red - salida_real)**2

def derivada_ecm(prediccion_red, salida_real):
    return 2 * (prediccion_red - salida_real) * 1

def feedforwad(entradas, pesos, salida, biases):
    # Capa Oculta:
    suma_ponderada_capa_oculta = np.dot(entradas, pesos[0]) + biases[0]
    neuronas_capa_oculta_valor_activacion = sigmoide(suma_ponderada_capa_oculta)
    # Capa de Salida o Capa 2:
    suma_ponderada_capa_salida = np.dot(neuronas_capa_oculta_valor_activacion, pesos[1]) + biases[1]
    prediccion_de_la_red = sigmoide(suma_ponderada_capa_salida)
    # Error de la red:
    ECM = error_cuadratico_medio(salida, prediccion_de_la_red)

    valores_de_activacion = [neuronas_capa_oculta_valor_activacion, prediccion_de_la_red]
    return valores_de_activacion, ECM

def backpropagation(pesos, biases, activaciones, salida_real):
    delta_neurona_salida = derivada_ecm(activaciones[-1], salida_real) * derivada_sigmoide(activaciones[-1])  #  (1,1)
    deltas_neuronas_capa_oculta = derivada_sigmoide(activaciones[0]).T * (pesos[-1]@delta_neurona_salida)  #  (3,1)

    Deltas = [delta_neurona_salida, deltas_neuronas_capa_oculta]
    Deltas.reverse()  # [deltas_capa_oculta, delta_neurona_SALIDA]
    return Deltas

def actualizar_params(TA, entradas, activaciones, deltas, pesos, biases):
    pesos_gradientes_capa_de_salida = activaciones[0].T * deltas[-1]
    pesos_gradientes_capa_oculta = entradas.T@(deltas[0].T)
    # Una vez obtenemos los pesos gradientes de cada capa, procedemos a actualizar los parámetros:
    pesos = [pesos_capa - TA*gradiente for pesos_capa,gradiente in zip(pesos, [pesos_gradientes_capa_oculta, pesos_gradientes_capa_de_salida])]
    biases = [biases_x_capa - TA*delta.T for biases_x_capa,delta in zip(biases,deltas)]

    return pesos, biases

def graficar_entrenamiento(epocas, errores):
    f = plt.figure()
    f.set_figwidth(15)

    X_axis = range(1, epocas+1)
    Y_axis = errores
    plt.plot(X_axis, Y_axis)
    plt.xticks(X_axis)
    plt.xlabel("Épocas")
    plt.ylabel("Error")
    plt.show()  # Nunca olvides agregar "plt.show()" siempre que estes usando matplotlib en VS code.

# ENTRENAMIENTO:
pesos_a_actualizar, biases_a_actualizar = deepcopy(Matrices_Pesos), deepcopy(Matrices_Biases)

errores_por_epoca = list()
TA = 0.98
epocas = 80
numero_de_epoca = 1
while numero_de_epoca <= epocas:
    print("\nÉpoca: ", numero_de_epoca)
    predicciones, error = feedforwad(entradas, pesos_a_actualizar, salida_real, biases_a_actualizar)
    deltas = backpropagation(pesos_a_actualizar, biases_a_actualizar, predicciones, salida_real)
    pesos_a_actualizar, biases_a_actualizar = actualizar_params(TA, entradas, predicciones, deltas, pesos_a_actualizar, biases_a_actualizar)

    print("Predicción de la Red: ", predicciones[-1])
    print("Error de la Red: ", error)
    errores_por_epoca.append(error.reshape(-1))

    numero_de_epoca += 1

graficar_entrenamiento(epocas, errores_por_epoca)