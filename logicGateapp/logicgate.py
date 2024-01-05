import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import *

st.title("Neural Network Implementation of logic gates")

plt.style.use("ggplot")
neuronsInHiddenLayers = 4

@st.cache(suppress_st_warning=True)
def train_model(epoch, alpha, X, Y):
    
    inputFeatures = X.shape[0]
    outputFeatures = Y.shape[0]
    parameters = initializeParameters(inputFeatures, neuronsInHiddenLayers, outputFeatures)
    epoch = epoch
    alpha = alpha
    losses = np.zeros((epoch, 1))
    for i in range(epoch):
        losses[i, 0], cache, A2 = forwardPropagation(X, Y, parameters)
        gradients = backwardPropagation(X, Y, cache)
        parameters = updateParameters(parameters, gradients, alpha)
    return losses


st.cache()
def test_result():
    st.sidebar.subheader("Select inputs and submit")
    # inpt1 = []
    # inpt2 = []
    # with st.sidebar.form(key='columns_in_form'):
    #     cols = st.columns(4)
    #     for row in range(0, 2):
    #         print(row)
    #         for i, col in enumerate(cols):
    #             inpt1.append(int(col.selectbox(f'', ['1', '0'], key=i)))
    #         print("")
    #     submitted = st.form_submit_button('Submit')
    form = st.sidebar.form(key='my_form')
    inpt1 = [int(ch) for ch in form.text_input(label='Enter first array', max_chars=4, placeholder="only 1's and 0's")]
    inpt2 = [ int(ch) for ch in form.text_input(label='Enter second array', max_chars=4, placeholder="only 1's and 0's")]
    submit_button = form.form_submit_button(label='Submit')
    if submit_button:
        if inpt1 == [] or inpt2 == []:
            st.warning("Empty values given to one or more input field")
        else:
            # st.write(str(inpt1))
            # st.write(str(inpt2))
            
            # st.write(data)
            final_input = np.array([[1, 0, 0, 1],
                        [0, 1, 1, 0]])
            Y = np.array([[0, 0, 0, 0]])
            inputFeatures = final_input.shape[0]
            outputFeatures = Y.shape[0]
            parameters = initializeParameters(inputFeatures, neuronsInHiddenLayers, outputFeatures)
            cost, _, A2 = forwardPropagation(final_input, Y, parameters)
            prediction = (A2 > 0.5) * 1.0
            data = {'X':["[" + str(inpt1[i]) + "," + str(inpt2[i]) + "]" for i in range(len(inpt2))],
                'Predicted output':prediction[0]}
            st.subheader("Test result")
            df = pd.DataFrame(data)
            st.write(df)
            print(prediction)
            st.write(str(prediction))

# @st.cache()
def evaluate_performance(losses):
    st.subheader("Trained Model's Performance")
    fig2 = plt.figure(figsize=(16,10))
    # plt.figure()
    # plt.legend()
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    # plt.show()
    st.pyplot(fig2)


def main():
    gates = ('AND Gate', 'OR Gate', 'XOR Gate', 'NAND Gate', 'NOR Gate')
    st.sidebar.header("Set parameters")
    # neuronsInHiddenLayers = st.sidebar.number_input("Number of neurons in hidden layers", 2, 4)
    act_func = st.sidebar.selectbox("Select activation funciton", ('sigmoid', 'ReLu'))
    alpha = st.sidebar.slider("Select learning rate(alpha)", 0.01, 0.05, step=0.01)
    epoch = st.sidebar.slider("How many epochs?", 10000, 100000, step=10000)
    selected_gate = st.sidebar.selectbox("Select a logic gate", gates, index=gates.index('AND Gate'))
    if selected_gate == 'AND Gate':
        # input features
        X = np.array([[0, 0, 1, 1],
                    [0, 1, 0, 1]])
        Y = np.array([[0, 0, 0, 1]])

        data = {'X':['[0, 0]','[0, 1]', '[1, 0]', '[1, 1]'],
                'Y':[0, 0, 0, 1]}
        st.subheader("AND Gate Truth Table")
        df = pd.DataFrame(data)
        st.write(df)
        
        # st.sidebar.button("Train Model", on_click=
        losses = train_model(epoch, alpha, X, Y)
            # if st.sidebar.button("Evaluate Model's Performance"):
        st.sidebar.button("Evaluate performance", on_click=evaluate_performance(losses))
        # if st.sidebar.button("Test w  ith Inputs"): 
        test_result()

    
    # if selected_gate == 'NAND Gate':
    #     # input features
    #     X = np.array([[0, 0, 1, 1],
    #                 [0, 1, 0, 1]])
    #     Y = np.array([[1, 1, 1, 0]])

    #     data = {'X':['[0, \t0]','[0, 1]', '[1, 0]', '[1, 1]'],
    #             'Y':[1, 1, 1, 0]}
    #     st.subheader("NAND Gate Truth Table")
    #     df = pd.DataFrame(data)
    #     st.write(df)

    # if selected_gate == 'OR Gate':
    #     # input features
    #     X = np.array([[0, 0, 1, 1],
    #                 [0, 1, 0, 1]])
    #     Y = np.array([[0, 1, 1, 1]])

    #     data = {'X':['[0, \t0]','[0, 1]', '[1, 0]', '[1, 1]'],
    #             'Y':[0, 1, 1, 1]}
    #     st.subheader("OR Gate Truth Table")
    #     df = pd.DataFrame(data)
    #     st.write(df)
    
    # if selected_gate == 'XOR Gate':
    #     # input features
    #     X = np.array([[0, 0, 1, 1],
    #                 [0, 1, 0, 1]])
    #     Y = np.array([[0, 1, 1, 0]])

    #     data = {'X':['[0, \t0]','[0, 1]', '[1, 0]', '[1, 1]'],
    #             'Y':[0, 1, 1, 0]}
    #     st.subheader("XOR Gate Truth Table")
    #     df = pd.DataFrame(data)
    #     st.write(df)
    
    # if selected_gate == 'NOR Gate':
    #     # input features
    #     X = np.array([[0, 0, 1, 1],
    #                 [0, 1, 0, 1]])
    #     Y = np.array([[1, 0, 0, 0]])

    #     data = {'X':['[0, \t0]','[0, 1]', '[1, 0]', '[1, 1]'],
    #             'Y':[1, 0, 0, 0]}
    #     st.subheader("NOR Gate Truth Table")
    #     df = pd.DataFrame(data)
    #     st.write(df)
    
    




if __name__=='__main__':
    main()