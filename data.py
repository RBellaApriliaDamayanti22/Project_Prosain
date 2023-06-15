import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib as plt
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

st.title("Penambangan Data")
st.write("### Dosen Pengampu: Mula'ab, S.Si., M.Kom.")
st.write("##### Kelompok")
st.write("##### Fadetul Fitriyeh- 200411100189")
st.write("##### R.Bella Aprilia Damayanti - 200411100082")

upload_data, persiapan_data, preprocessing, modeling, implementation = st.tabs(["Upload Data", "Persiapan Data","Preprocessing", "Modeling", "Implementation"])

with upload_data:
    st.write('Dataset yang digunakan yaitu dataset XL AXIATA yang diambil dari yahoo.com')
    data = pd.read_csv('https://raw.githubusercontent.com/RBellaApriliaDamayanti22/Project_Prosain/main/XL%20axiata.csv')
    st.dataframe(data)
    st.write("Penjelasan Nama - Nama Kolom : ")
    st.write("""
    <ol>
    <li>Date (Tanggal): Tanggal dalam data time series mengacu pada tanggal tertentu saat data keuangan dikumpulkan atau dilaporkan. Ini adalah waktu kapan data keuangan yang terkait dengan PT Adaro Minerals Indonesia dicatat.</li>
    <li>Open (Harga Pembukaan): Harga pembukaan adalah harga perdagangan PT Adaro Minerals Indonesia pada awal periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga pembukaan menunjukkan harga perdagangan pertama dari PT Adaro Minerals Indonesia pada periode tersebut.</li>
    <li>High (Harga Tertinggi): Harga tertinggi adalah harga tertinggi yang dicapai oleh PT Adaro Minerals Indonesia selama periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga tertinggi mencerminkan harga perdagangan tertinggi yang dicapai oleh PT Adaro Minerals Indonesia dalam periode tersebut.</li>
    <li>Low (Harga Terendah): Harga terendah adalah harga terendah yang dicapai oleh PT Adaro Minerals Indonesia selama periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga terendah mencerminkan harga perdagangan terendah yang dicapai oleh PT Adaro Minerals Indonesia dalam periode tersebut.</li>
    <li>Close (Harga Penutupan): Harga penutupan adalah harga terakhir dari PT Adaro Minerals Indonesia pada akhir periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Harga penutupan menunjukkan harga terakhir di mana PT Adaro Minerals Indonesia diperdagangkan pada periode tersebut.</li>
    <li>Adj Close (Harga Penutupan yang Disesuaikan): Adj Close, atau harga penutupan yang disesuaikan, adalah harga penutupan yang telah disesuaikan untuk faktor-faktor seperti dividen, pemecahan saham, atau perubahan lainnya yang mempengaruhi harga saham PT Adaro Minerals Indonesia. Ini memberikan gambaran yang lebih akurat tentang kinerja saham dari waktu ke waktu karena menghilangkan efek dari perubahan-perubahan tersebut.</li>
    <li>Volume: Volume dalam konteks data keuangan PT Adaro Minerals Indonesia mengacu pada jumlah saham yang diperdagangkan selama periode waktu tertentu, seperti hari perdagangan atau sesi perdagangan. Volume mencerminkan seberapa aktifnya perdagangan saham PT Adaro Minerals Indonesia dalam periode tersebut.</li>
    </ol>
    """,unsafe_allow_html=True)
with persiapan_data:
    # memebersihkan data yang 0
    def clean(data):
        vol = data["Volume"].values
        clean = []
        for i in range (len(vol)):
            if vol[i]==0:
                data = data.drop([i])
        
        return data
    # split a univariate sequence into samples
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    
     # split a univariate sequence into samples
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    
    inp_param = 6
    data_clean = clean(data)
    volume = data_clean['Volume'].values
    X, y = split_sequence(volume, inp_param)

    # column names to X and y data frames
    df_X = pd.DataFrame(X, columns=['input-'+str(i+1) for i in range(inp_param-1, -1,-1)])
    df_y = pd.DataFrame(y, columns=['output'])

    # concat df_X and df_y
    df = pd.concat([df_X, df_y], axis=1)

    st.write(df)

with preprocessing :
    # Split data
    training = pd.DataFrame(df.iloc[:359, :].values)
    test = pd.DataFrame(df.iloc[359:, :].values)

    scaler= MinMaxScaler()
    training_x = training.iloc[:, 0:6]
    training_y = training.iloc[:, 6:]

    X_norm= scaler.fit_transform(training_x)
    st.write(X_norm)
    # y_norm= scaler.fit_transform(df_y)

    X_norm= scaler.fit_transform(training_x)



with modeling :
    #X_norm, x_test, training_y, y_test = train_test_split(X_norm, test_size=0.2, random_state=1)
    x_test = test.iloc[:, :6]
    y_test = test.iloc[:, 6:]
    y_test.set_axis(["y_test"], axis="columns")

    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighbors')
        destree = st.checkbox('Decision Tree')
        submitted = st.form_submit_button("Submit")

        if naive:
            model_n = GaussianNB()
            model_n.fit(X_norm, training_y)
            y_pred3=model_n.predict(x_test)
            #gaussian_accuracy = round(100 * accuracy_score(y_test, y_pred3), 2)
            from sklearn.metrics import mean_absolute_percentage_error
            mape = mean_absolute_percentage_error(y_test, y_pred3)
            #st.write('Model Gaussian Naive Bayes accuracy score:', gaussian_accuracy)
            st.write('MAPE Model Gaussian Naive Bayes:', mape)

        if k_nn:
            # import knn
            from sklearn.neighbors import KNeighborsRegressor
            model_knn = KNeighborsRegressor(n_neighbors=30)
            model_knn.fit(X_norm, training_y)
            y_pred2=model_knn.predict(x_test)
            # knn_accuracy = round(100 * accuracy_score(y_test, y_pred2), 2)    
            from sklearn.metrics import mean_absolute_percentage_error
            mape = mean_absolute_percentage_error(y_test, y_pred2)
            # st.write("Model K-Nearest Neighbors accuracy score:", knn_accuracy )
            st.write('MAPE Model K-Nearest Neighbors :', mape)

        if destree:
            #klasifikasi menggunakan decision tree
            model_tree = tree.DecisionTreeClassifier(random_state=3, max_depth=1)
            model_tree.fit(X_norm, training_y)
            y_pred1=model_tree.predict(x_test)
            #dt_accuracy = round(100 * accuracy_score(y_test, y_pred1), 2)
            from sklearn.metrics import mean_absolute_percentage_error 
            mape = mean_absolute_percentage_error(y_test, y_pred1) 
            #st.write("Model Decision Tree accuracy score:", dt_accuracy)
            st.write('MAPE Model Decision Tree:', mape)

with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        input1 = st.number_input('input 1:')
        input2 = st.number_input('input 2:')
        input3 = st.number_input('input 3:')
        input4 = st.number_input('input 4:')
        input5 = st.number_input('input 5:')
        input6 = st.number_input('input6:')
        # model = st.selectbox('Pilih model untuk prediksi:',
        #                     ('Gaussian Naive Bayes', 'K-Nearest Neighbors', 'Decision Tree'))
        submitted = st.form_submit_button("Submit")

        if submitted:
            input_data = np.array([[input1,input2,input3,input4,input5,input6]])
            scaler = MinMaxScaler()
            scaled_input = scaler.fit_transform(input_data)
            #nama_model = 'model_knn_pkl' 
            # model == 'Decision Tree'
            mod = tree.DecisionTreeClassifier()
            mod.fit(X_norm, training_y)
            #nama_model = "mpl_regs"y

            # load the model from disk
            # loaded_model = pickle.load(open(nama_model, 'rb'))
            # input_pred = loaded_model.predict(scaled_input)

            # st.subheader('Hasil Prediksi')
            # st.write('Menggunakan Model:', model)
            # st.write('Volume:', input_pred)
            input_pred = mod.predict(scaled_input)

            # st.subheader('Hasil Prediksi')
            # st.write('Menggunakan Model:', model)
            st.success(f'hasil prediksi: {input_pred.reshape(1)[0]}')
