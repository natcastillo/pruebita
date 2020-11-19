import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

st.title("LABORATORIO #1 :hankey:")

#Sidebar
    #GENERACION DE LAS SEÑALES
st.sidebar.title('Generación de Señales')
fun=st.sidebar.selectbox('Seleccione el tipo de señal', ['Senoidal', 'Cuadrática', 'Lineal', 'Exponencial', 'Pulso','Triangular','Cuadrada', 'Pulse Train'])
dis = st.sidebar.checkbox('Tiempo discreto')
if dis:
   s = st.sidebar.slider('Número de pasos', 25,100,30)

if fun == 'Senoidal':
    st.header('Señal Senoidal')
    st.markdown('De la forma **A*Sen(ω·t)*')
    
    if dis:
        x = np.linspace(0, 2*np.pi, s)
    else:
        x = np.arange(0, 2*np.pi, 0.01)
    a = st.sidebar.number_input('Digite el valor de la amplitud (A)', -10.0, 10.0, 1.0)
    w = st.sidebar.number_input('Digite el valor de la frecuencia (ω)', -5, 15, 1)
    y=a*np.sin(w*x)
    k=0

elif fun == 'Cuadrática':
    st.header('Señal Cuadrática')
    st.markdown('De la forma **a·t^2 + b·t + c**')
    
    if dis:
        x = np.linspace(-np.pi, np.pi, s)
    else:
        x = np.arange(-np.pi, np.pi, 0.01)
    a = st.sidebar.number_input('Digite el valor de a', -15, 15, 1)
    b = st.sidebar.number_input('Digite el valor de b', -15, 15, 0)
    c = st.sidebar.number_input('Digite el valor de c', -15, 15, 0)
    y=a*(x**2)+b*x+c
    k=1

elif fun == 'Lineal':
    st.header('Señal Lineal')
    st.markdown('De la forma **m·x + b**')

    if dis:
        x = np.linspace(-np.pi, np.pi, s)
    else:
        x = np.arange(-np.pi, np.pi, 0.01)
    m = st.sidebar.number_input('Digite el valor de m', -15, 15, 1)
    b = st.sidebar.number_input('Digite el valor de b', -15, 15, 0)
    y=m*x+b
    k=1

elif fun == 'Exponencial':
    st.header('Señal Exponencial')
    st.markdown('De la forma **A·e^b·t**')

    if dis:
        x = np.linspace(0, 2*np.pi, s)
    else:
        x = np.arange(0, 2*np.pi, 0.01)
    a = st.sidebar.number_input('Digite el valor de A', -15, 15, 1)
    b = st.sidebar.number_input('Digite el valor de b', -15, 15, 1)
    y=a*np.exp(-b*x)
    k=0

elif fun == 'Pulso':
    st.header('Señal Pulso')
    L = st.sidebar.number_input('Digite el Ancho del pulso', 1, 15)
    a = st.sidebar.number_input('Digite la Amplitud del pulso', 1.0, 15.0)

    if dis:
        x = np.linspace(-L/2-3,L/2+3, s)
    else:    
        x = np.arange (-L/2-3,L/2+3, 0.01)
    y=x*0
    y[(-L/2<x) &(x< L/2 )] = a
    k=2

elif fun == 'Triangular':
    st.header('Señal Triangular')
    a = st.sidebar.number_input('Digite la amplitud (A)', 0.1,10.0,1.0)
    w = st.sidebar.number_input('Digite la frecuencia (ω)', 1, 15,3)
    
    if dis:
        x = np.linspace(0, 2*np.pi, s)
    else:       
        x = np.arange(0, 2*np.pi, 0.01)
    y= (signal.sawtooth(w * x))*a
    k=0
    
elif fun == 'Cuadrada':
    st.header('Señal Cuadrada')
    a = st.sidebar.number_input('Digite la amplitud (A)', 0.1,10.0,1.0)
    w = st.sidebar.number_input('Digite la frecuencia (ω)', 1, 15,3)

    if dis:
        x = np.linspace(0, 2*np.pi, s)
    else:       
        x = np.arange(0, 2*np.pi, 0.01)
    y= (signal.square(w * x))*a
    k=0

elif fun == 'Pulse Train':
    st.header('Secuencia de impulsos')
    x = st.sidebar.text_input('Digite los valores de t, separados por coma ( , )',0)
    y = st.sidebar.text_input('Digite la amplitud de cada valor t correspodiente, separados por coma ( , )',0)
    x = np.array(np.matrix(x)).ravel()
    y = np.array(np.matrix(y)).ravel()
    k=3

#plot
    #Representación gráfica de las señales4
plt.figure(figsize=(10,4.5))
if (k==3):
    plot = st.sidebar.button('graficar')
    if plot:
        g=plt.stem(x,y, 'b-.', markerfmt = 'D')
elif dis:
    plt.stem(x,y, 'b:', markerfmt = 'D')
else:
    g=plt.plot(x,y, lw=3)

    #Estética de las gráficas
plt.ylabel("Amplitude", fontsize='x-large')
plt.xlabel("t", fontsize='xx-large')
plt.subplot().axhline(0, color='black', lw=1)
plt.subplot().axvline(0, color='black', lw=1)
plt.subplot().grid(linewidth =0.4)


#OPERACION CON LAS SEÑALES
    #Sidebar
st.sidebar.title('Selección operación')
op = st.sidebar.selectbox('Options', ['Escalamiento en el tiempo','Desplazamiento en el tiempo','Escalamiento de amplitud'])

if op == 'Escalamiento en el tiempo':
   A = st.sidebar.number_input('Seleccione A, donde g(x) = y(Ax)', 0.1, 15.0, 2.0)
elif op == 'Desplazamiento en el tiempo':
   A = st.sidebar.number_input('Seleccione A, donde g(x) = y(X+A)', -15.0, 15.0, -1.0)
elif op == 'Escalamiento de amplitud':
   A = st.sidebar.number_input('Seleccione A, donde g(x) = A*y(x)', 0.1, 15.0, 2.0)
   z = A*y

animation = st.sidebar.button('Animar')
graph = st.empty() 

if animation:
   frames = 10
   if op == 'Escalamiento en el tiempo':
      st.write('Tiempo escalado por: \n', A)
      for i in range (frames):
            #Grafica animada de la operación
            if k==3:
                plt.stem(x,y, 'b-', markerfmt = 'D', label='Señal original')
                plt.stem(x/(1+i*(A-1)/(frames-1)),y, 'r-.', markerfmt = 'D', label='Señal original')
            else:
                plt.plot ( x, y, lw=2.5, label = 'Señal Original')
                plt.plot(x/(1+i*(A-1)/(frames-1)), y, lw=2.5, label = 'Escalada en el tiempo')

            #Limites de gráfica
            if (A<1)&(k==0):
                plt.xlim(min(x)-0.3, max(x)/A + 0.3)
            elif (k==1)&(A<1):
                plt.xlim(-np.pi/A - 0.5, np.pi/A + 0.5)
            elif ((k==2)&(A<1)):
                plt.xlim(-(L+2)/(2*A),(L+2) /(2*A))
            elif(k==3):
                if (min(x)>0)&(A>1):
                    plt.xlim(-0.5,max(x)+0.5)
                elif (min(x)>0)&(A<1):
                    plt.xlim(-0.5,max(x)/A + 0.5)
                elif (min(x)<0)&(A>1):
                    plt.xlim(-0.5,max(x)+0.5)
                elif (min(x)>0)&(A>1):
                    plt.xlim(-0.5,max(x)+0.5)
            else:
                plt.xlim(min(x)-0.5, max(x)+0.5)

            #Estética de la gráfica de la animación
            plt.legend (loc = "best")
            plt.ylabel("Amplitud", fontsize='x-large')
            plt.xlabel("t", fontsize='xx-large')
            plt.subplot().grid(linewidth =0.4)
            plt.subplot().axhline(0, color='black', lw=1)
            plt.subplot().axvline(0, color='black', lw=1)
          
            graph.pyplot()
         
   elif op == 'Desplazamiento en el tiempo':
      st.write('Tiempo desplazado por: ', A)
      for i in range (frames):
            #Grafica animada de la operación
            if k==3:
                plt.stem(x,y, 'b-', markerfmt = 'D', label='Señal original')
                plt.stem(x - A*(i)/(frames-1),y, 'r-.', markerfmt = 'D', label='Desplazada en el tiempo')
            else:
                plt.plot ( x, y, lw=2.5, label = 'Señal original')
                plt.plot((x - A*(i)/(frames-1)), y, lw=2.5, label = 'Desplazada en el tiempo')

            #Límites de la gráfica
            if A>0:
                plt.xlim(min(x)-A-0.5,max(x)+0.5)
            elif A<0:
                plt.xlim(min(x)-0.3, max(x)-A+0.3)            

            #Estética de la gráfica de la animación
            plt.legend (loc = "best")
            plt.ylabel("Amplitud", fontsize='x-large')
            plt.xlabel("t", fontsize='xx-large')
            plt.subplot().grid(linewidth =0.5)
            plt.subplot().axhline(0, color='black', lw=1)
            plt.subplot().axvline(0, color='black', lw=1)
            
            graph.pyplot()

   elif op == 'Escalamiento de amplitud':
      st.write('Amplitud escalada por: ', A)
      for i in range (frames):
            #Grafica animada de la operación
            if k==3:
                plt.stem(x,y, 'b-', markerfmt = 'D', label='Señal original')
                plt.stem(x,y*(1+i*(A-1)/(frames-1)), 'r-.', markerfmt = 'D', label='Desplazada en el tiempo')
            else:
                plt.plot ( x, y, lw=2.5, label = ' Señal original')
                plt.plot(x, y*(1+i*(A-1)/(frames-1)), lw=2.5, label = 'Escalada en Amplitud')

            #Límites de la gráfica
            if k==2:            
                if (A>1):
                    plt.ylim(min(z), max(z)+0.3)
                else:
                    plt.ylim(min(y), max(y)+0.2)
            elif k==3:
                plt.ylim(-0.3,max(z)+0.3)
            else:
                if (A>1):
                    plt.ylim(min(z)-0.2, max(z)+0.2)
                else:
                    plt.ylim(min(y)-0.2, max(y)+0.2)
            
            #Estética de la gráfica de la animación
            plt.legend (loc = "best")
            plt.ylabel("Amplitud", fontsize='x-large')
            plt.xlabel("t", fontsize='xx-large')             
            plt.subplot().grid(linewidth =0.4)
            plt.subplot().axhline(0, color='black', lw=1)
            plt.subplot().axvline(0, color='black', lw=1)

            graph.pyplot()

st.pyplot(clear_figure=True) #Este código hace que la gráfica apareza en el Streamlit