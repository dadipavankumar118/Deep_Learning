import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import io
from io import StringIO

import sklearn
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import SGD

from mlxtend.plotting import plot_decision_regions

# Function to create neural network graph
def create_neural_network_graph(num_inputs, num_outputs, hidden_layers, neurons_per_layer):
    G = nx.DiGraph()
    
    # Add input layer
    for i in range(num_inputs):
        G.add_node(f'Input {i+1}', layer=0)
    
    # Add hidden layers
    for layer in range(hidden_layers):
        for neuron in range(neurons_per_layer[layer]):
            G.add_node(f'HL {layer+1} N {neuron+1}', layer=layer+1)
            if layer == 0:
                for input_neuron in range(num_inputs):
                    G.add_edge(f'Input {input_neuron+1}', f'HL {layer+1} N {neuron+1}')
            else:
                for prev_neuron in range(neurons_per_layer[layer-1]):
                    G.add_edge(f'HL {layer} N {prev_neuron+1}', f'HL {layer+1} N {neuron+1}')
    
    # Add output layer
    for neuron in range(num_outputs):
        G.add_node(f'Output {neuron+1}', layer=hidden_layers+1)
        for prev_neuron in range(neurons_per_layer[-1]):
            G.add_edge(f'HL {hidden_layers} N {prev_neuron+1}', f'Output {neuron+1}')
    
    return G

# Function to draw neural network graph
def draw_neural_network(G, figsize=(10, 8)):
    pos = nx.multipartite_layout(G, subset_key='layer')
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)
    ax.set_title("Neural Network Graph")

    # Save the plot as an image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    
    return img_buffer



def page_1_function():
    # Sidebar inputs
    st.title('Create a Random DataSet.')
    st.sidebar.subheader("DataSet Configuration")
    n_datapoints = st.sidebar.number_input("Enter number of data points", min_value=100, value=1000)
    cls_sep = st.sidebar.slider('Select Class separation between each Class', min_value=0.0, max_value=4.0, step=0.01, value=1.0)
    rand = st.sidebar.checkbox('Random State on', value=False)
    if rand:
        random = None
    else:
        random = 18
    test_data_size = st.sidebar.slider('Select size for the test data', min_value=0.0, max_value=1.0, step=0.01, value=0.25)

        # Generate synthetic dataset
    feature_var, class_var = make_classification(n_samples=n_datapoints, n_features=2, n_informative=2,
                                                    n_redundant=0, n_repeated=0, n_classes=2,
                                                    n_clusters_per_class=1, class_sep=cls_sep, random_state=random)


        # Data preprocessing
    x_train, x_test, y_train, y_test = train_test_split(feature_var, class_var, test_size=test_data_size,
                                                            random_state=random, stratify=class_var)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

        # Model Configuration
    st.sidebar.subheader("Model Configuration")
    num_inputs = x_train.shape[1]
    num_outputs = 1
    hidden_layers = st.sidebar.slider("Enter number of hidden layers", min_value=1, max_value=7, value=2)

    neurons_per_layer = []
    for i in range(hidden_layers):
        neurons = st.sidebar.slider(f"Enter number of neurons for hidden layer {i+1}", min_value=1, max_value=8, value=2)
        neurons_per_layer.append(neurons)

    act_func = st.sidebar.selectbox('Select the Activation Function', ('tanh', 'sigmoid'))
    use_bias = st.sidebar.checkbox('Use Bias', value=True)


        # Model Training
    st.sidebar.subheader("Model Training")
    lr = st.sidebar.selectbox('Choose the learning rate', (0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10))
    epoch = st.sidebar.number_input('Enter number of Epochs', min_value=1, value=8)
    batchSize = st.sidebar.number_input('Select the Batch Size', min_value=1, max_value=n_datapoints, value=n_datapoints // 10)
    val_split = st.sidebar.slider('Select size for the Validation data', min_value=0.0, max_value=1.0, step=0.01, value=0.2)

    if st.sidebar.button('Submit'):

        st.title('Scatterplot for the data')
        fig, ax = plt.subplots()
        sns.scatterplot(x=feature_var[:, 0], y=feature_var[:, 1], hue=class_var)
        st.pyplot(fig)


        st.title("Neural Network Graph")
        G = create_neural_network_graph(num_inputs, num_outputs, hidden_layers, neurons_per_layer)
        st.image(draw_neural_network(G, figsize=(20, 10)).getvalue())


            # Model compilation
        sgd = SGD(learning_rate=lr)
        model = Sequential()
        model.add(InputLayer(shape=(x_train.shape[1],)))
        for i in range(hidden_layers):
            model.add(Dense(neurons_per_layer[i], activation=act_func, use_bias=use_bias))
        model.add(Dense(1, activation='sigmoid', use_bias=use_bias))
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

            # Model training
        hist = model.fit(x_train, y_train, epochs=epoch, batch_size=batchSize, validation_split=val_split)

        # Plot Training and Validation Loss
        st.title('Training and Validation Loss')
        fig_loss, ax_loss = plt.subplots()
        plt.plot(range(1,epoch+1), hist.history['loss'], label= 'Training Loss')
        plt.plot(range(1,epoch+1), hist.history['val_loss'], label= 'Validation Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Training and Validation Loss')
        ax_loss.legend()
        st.pyplot(fig_loss)

            # Plot decision regions
        st.title('Decision Regions')
        fig, ax = plt.subplots()
        plot_decision_regions(x_test, y_test, clf=model, ax=ax)
        st.pyplot(fig)

        test_loss, test_accu = model.evaluate(x_test,y_test)
        train_loss, train_accu = model.evaluate(x_train, y_train)

        font_size = 24

        test_result = 'Test accuracy: '+ str(round(test_accu,2))
        train_result = 'Train accuracy: '+ str(round(train_accu,2))

        text_color1 = 'green'
        text_color2 = 'blue'

        st.markdown(f""" <div style="display: flex; flex-direction: column;
                        justify-content: center; align-items: center;">
                    <p style="font-size:{font_size}px; color:{text_color1};">{test_result}</p>
                    <p style="font-size:{font_size}px; color:{text_color2};">{train_result}</p>
                    </div>""",unsafe_allow_html=True)    