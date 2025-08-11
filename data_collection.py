# Imports for P300
import multiprocessing
import time
from multiprocessing import Process, Value
import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Style
import scipy.stats as stats

# Imports for OpenBCI
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, WindowFunctions, DetrendOperations
import csv

# Imports from Empatica
import socket
import time
import pylsl
import numpy as np
import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from scipy.fft import rfft, rfftfreq

# # Create a Value data object # #
# This object can store a single integer and share it across multiple parallel
# processes
seconds = Value("i", 0)
counts = Value("i", 0)

tiempo_prueba = 570 # segundos
            
# This function is the countup timer, the count is set to 0 before the script
# waits for a second, otherwise the beep will sound several times before the second 
# changes.
def timer(second, count, timestamps):
    # First we initialize a variable that will contain the moment the timer began and 
    # we store this in the timestamps list that will be stored in a CSV.    
    time_start = time.time()
    timestamps.append(time_start)
    while True:
        # The .get_lock() function is necessary since it ensures they are 
        # sincronized between both functions, since they both access to the same 
        # variables
        with second.get_lock(), count.get_lock():
            # We now calculate the time elappsed between start and now. 
            # (should be approx. 1 second)
            second.value = int(time.time() - time_start)
            count.value = 0
            if(second.value == tiempo_prueba):
                return
            print(second.value, end="\r")
        # Once we stored all the info and make the calculations, we sleep the script for
        # one second. This is the magic of the script, it executes every  ~1 second.
        time.sleep(1) #0.996

def empatica(second, folder):
    global BVP_array, Acc_array, GSR_array, Temp_array, IBI_array
    global BVP_tuple, ACC_tuple, GSR_tuple, Temp_tuple, IBI_tuple
    global Temporal_BVP_array, Temporal_GSR_array, Temporal_Temp_array, Temporal_IBI_array, BVP_Graph_value
    global counter
    global x_BVP_val, x_GSR_val, x_Temp_val, x_IBI_val
    global y_BVP_val, y_GSR_val, y_Temp_val, y_IBI_val

    # VARIABLES USED TO STORE & GRAPH DATA
    BVP_array, Acc_array, GSR_array, Temp_array, IBI_array = [], [], [], [], []
    BVP_tuple, ACC_tuple, GSR_tuple, Temp_tuple, IBI_tuple = (), (), (), (), ()
    Temporal_BVP_array, Temporal_GSR_array, Temporal_Temp_array, Temporal_IBI_array = [], [], [], []
    BVP_Graph_value = None
    counter = 0 # Used to pop values from arrays to perform a "moving" graph.
    x_BVP_val, x_GSR_val, x_Temp_val, x_IBI_val = [], [], [], []
    y_BVP_val, y_GSR_val, y_Temp_val, y_IBI_val = [], [], [], []

    # SELECT DATA TO STREAM
    acc = True      # 3-axis acceleration
    bvp = True      # Blood Volume Pulse
    gsr = True      # Galvanic Skin Response (Electrodermal Activity)
    tmp = True      # Temperature
    ibi = True

    serverAddress = '127.0.0.1'  #'FW 2.1.0' #'127.0.0.1'
    serverPort = 28000 #28000 #4911
    bufferSize = 4096
    # The wristband with SN A027D2 worked here with deviceID 8839CD
    deviceID = '834acd' #'8839CD' #'1451CD' # 'A02088' #'A01FC2'

    # PC = Purchase code
    # ID = Device ID
    # SN = Serial number

    # N     PC       SN      ID
    # 1  1197161b  A01FC2  834acd
    # 2  1716f3b5  A04BEB  de6f5a
    # 3  d56bbc72  A027D2  8839cd

    # de6f5a
    def connect():
        global s
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)

        print("Connecting to server")
        s.connect((serverAddress, serverPort))
        print("Connected to server\n")

        print("Devices available:")
        s.send("device_list\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

        print("Connecting to device")
        s.send(("device_connect " + deviceID + "\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

        print("Pausing data receiving")
        s.send("pause ON\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
        
    connect()

    time.sleep(1)

    def suscribe_to_data():
        if acc:
            print("Suscribing to ACC")
            s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if bvp:
            print("Suscribing to BVP")
            s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if gsr:
            print("Suscribing to GSR")
            s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if tmp:
            print("Suscribing to Temp")
            s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        if ibi:
            print("Suscribing to Ibi")
            s.send(("device_subscribe " + 'ibi' + " ON\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))

        print("Resuming data receiving")
        s.send("pause OFF\r\n".encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    suscribe_to_data()

    def prepare_LSL_streaming():
        print("Starting LSL streaming")
        if acc:
            infoACC = pylsl.StreamInfo('acc','ACC',3,32,'int32','ACC-empatica_e4')
            global outletACC
            outletACC = pylsl.StreamOutlet(infoACC)
            
            print(outletACC)
        if bvp:
            infoBVP = pylsl.StreamInfo('bvp','BVP',1,64,'float32','BVP-empatica_e4')
            global outletBVP
            outletBVP = pylsl.StreamOutlet(infoBVP)
        if gsr:
            infoGSR = pylsl.StreamInfo('gsr','GSR',1,4,'float32','GSR-empatica_e4')
            global outletGSR
            outletGSR = pylsl.StreamOutlet(infoGSR)
        if tmp:
            infoTemp = pylsl.StreamInfo('tmp','Temp',1,4,'float32','Temp-empatica_e4')
            global outletTemp
            outletTemp = pylsl.StreamOutlet(infoTemp)
        if ibi:
            infoIbi = pylsl.StreamInfo('ibi','Ibi',1,2,'float32','IBI-empatica_e4')
            global outletIbi
            outletIbi = pylsl.StreamOutlet(infoIbi)
    prepare_LSL_streaming()

    time.sleep(1)

    def reconnect():
        print("Reconnecting...")
        connect()
        suscribe_to_data()
        stream()

    def stream():
        try:
            print("Streaming...")
            try:
                with second.get_lock():
                    # When the seconds reach 312, we exit the functions.
                    if(second.value == tiempo_prueba):
                        plt.close()
                        return
                response = s.recv(bufferSize).decode("utf-8")
                #print(response)
                if "connection lost to device" in response:
                    print(response.decode("utf-8"))
                    reconnect()
                samples = response.split("\n") #Variable "samples" contains all the information collected from the wristband.
                # print(samples)
                # We need to clean every temporal array before entering the for loop.
                global Temporal_BVP_array
                global Temporal_GSR_array
                global Temporal_Temp_array
                global Temporal_IBI_array
                global flag_Temp #We only want one value of the Temperature to reduce the final file size.
                flag_Temp = 0
                for i in range(len(samples)-1):
                    try:
                        stream_type = samples[i].split()[0]
                    except:
                        continue
                    #print(samples)
                    if (stream_type == "E4_Acc"):
                        global Acc_array
                        global ACC_tuple
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = [int(samples[i].split()[2].replace(',','.')), int(samples[i].split()[3].replace(',','.')), int(samples[i].split()[4].replace(',','.'))]
                        outletACC.push_sample(data, timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        #print(data)#Added in 02/12/20 to show values
                        ACC_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        Acc_array.append(ACC_tuple)
                    if stream_type == "E4_Bvp":
                        global BVP_tuple
                        global BVP_array
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        outletBVP.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        #print(data)
                        Temporal_BVP_array.append(data)
                        BVP_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        BVP_array.append(BVP_tuple)
                    if stream_type == "E4_Gsr":
                        global GSR_array
                        global GSR_tuple
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        outletGSR.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        #print(data)
                        Temporal_GSR_array.append(data)
                        GSR_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        GSR_array.append(GSR_tuple)
                    if stream_type == "E4_Temperature":
                        global Temp_array
                        global Temp_tuple
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        outletTemp.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        #print(data)
                        Temporal_Temp_array.append(data)
                        if flag_Temp == 0:
                            Temp_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), Temporal_Temp_array[0])
                            Temp_array.append(Temp_tuple)
                            flag_Temp = 1
                    if stream_type == "E4_Ibi":
                        global IBI_array
                        global IBI_tuple
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        outletIbi.push_sample([data], timestamp=timestamp)
                        timestamp = datetime.fromtimestamp(timestamp)
                        #print(data)
                        Temporal_IBI_array.append(data)
                        IBI_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                        IBI_array.append(IBI_tuple)
                # We get the mean of the temperature and append them to the final array.
                # Temp_tuple = (datetime.now().isoformat(), np.mean(Temporal_Temp_array))
                # Temp_array.append(Temp_tuple)

                # We pause the acquisition of signals for one second
                #time.sleep(3)
            except socket.timeout:
                print("Socket timeout")
                reconnect()
        except KeyboardInterrupt:
            """
            #Debugging print variables
            print(BVP_array)
            print("*********************************************")
            print()
            print(Acc_array)
            print("*********************************************")
            print()
            print(GSR_array)
            print("*********************************************")
            print()
            print(Temp_array)
            print()
            """
            #print("Disconnecting from device")
            #s.send("device_disconnect\r\n".encode())
            #s.close()
    #stream()

    # MATPLOTLIB'S FIGURE AND SUBPLOTS SETUP
    """
    Gridspec is a function that help's us organize the layout of the graphs,
    first we need to create a figure, then assign a gridspec to the figure.
    Finally create the subplots objects (ax's) assigning a format with gs (gridspec).
    """
    fig = plt.figure(constrained_layout = True)
    gs = fig.add_gridspec(5,1)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title("Temperature")
    ax2 = fig.add_subplot(gs[1,0])
    ax2.set_title("Electrodermal Activity")
    ax3 = fig.add_subplot(gs[2,0])
    ax3.set_title("Blood Volume Pulse")
    ax4 = fig.add_subplot(gs[3,0])
    ax4.set_title("IBI")
    ax5 = fig.add_subplot(gs[4,0])
    ax5.set_title("Fast Fourier Transform")

    # Animation function: this function will update the graph in real time,
    # in order for it to work properly, new data must be collected inside this function.
    def animate(frame):
        global BVP_array
        global GSR_array
        global Temp_array
        global IBI_array
        global Temporal_BVP_array
        global Temporal_GSR_array
        global Temporal_Temp_array
        global Temporal_IBI_array
        global counter
        stream() # This is the function that connects to the Empatica.
        
        #x_BVP_val = np.linspace(0,len(Temporal_BVP_array)-1,num= len(Temporal_BVP_array))
        #x_GSR_val = np.linspace(0,len(Temporal_GSR_array)-1,num= len(Temporal_GSR_array))
        #x_Temp_val = np.linspace(0,len(Temporal_Temp_array)-1,num= len(Temporal_Temp_array))
        #x_IBI_val = np.linspace(0,len(Temporal_IBI_array)-1,num= len(Temporal_IBI_array))
        
        x_BVP_val = np.arange(0.015625,((len(Temporal_BVP_array))*0.015625)+0.015625,0.015625)
        x_GSR_val = np.arange(0.25,((len(Temporal_GSR_array))*0.25)+0.25,0.25)
        x_Temp_val = np.linspace(0,len(Temporal_Temp_array)-1,num= len(Temporal_Temp_array))
        x_IBI_val = np.linspace(0,len(Temporal_IBI_array)-1,num= len(Temporal_IBI_array))
        
        X = rfft(Temporal_BVP_array)
        xf = rfftfreq(len(Temporal_BVP_array), 1/64)

        # GRAPHING ASSIGNMENT SECTION
        # First the previous data must be cleaned, then we plot the array with the updated info.
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax1.set_ylim(25,40) #We fixed the y-axis values to observe a better data representation.
        #ax2.set_ylim(0, 0.5)
        ax3.set_ylim(-150,150)
        ax4.set_ylim(0,1)
        ax1.set_title("Temperature")
        ax2.set_title("Electrodermal Activity")
        ax3.set_title("Blood Volume Pulse")
        ax4.set_title("IBI")
        ax5.set_title("Fast Fourier Transform")
        ax1.set_ylabel("Celsius (°C)")
        ax2.set_ylabel("Microsiemens (µS)")
        ax3.set_ylabel("Nano Watt")
        ax4.set_ylabel("Seconds (s)")
        ax5.set_ylabel("Magnitude")
        ax1.set_xlabel("Samples")
        ax2.set_xlabel("Seconds")
        ax3.set_xlabel("Seconds")
        ax4.set_xlabel("Samples")
        ax5.set_xlabel("Frequency (Hz)")

        if (counter >= 2400):
            ax1.plot(x_Temp_val,Temporal_Temp_array, color = "#F1C40F")
            ax2.plot(x_GSR_val[-200:],Temporal_GSR_array[-200:], color = "#16A085")
            ax3.plot(x_BVP_val[-2000:],Temporal_BVP_array[-2000:])
            ax4.plot(x_IBI_val, Temporal_IBI_array, color = '#F2220C')
            ax5.plot(xf, np.abs(X))

        else:
            ax1.plot(x_Temp_val,Temporal_Temp_array, color = "#F1C40F")
            ax2.plot(x_GSR_val,Temporal_GSR_array, color = "#16A085")
            ax3.plot(x_BVP_val,Temporal_BVP_array)
            ax4.plot(x_IBI_val, Temporal_IBI_array, color = '#F2220C')
            ax5.plot(xf, np.abs(X))

        counter += 60  

    # Here es where the animation is executed. Try encaspsulation allows us
    # to stop the code anytime with Ctrl+C.
    try:
        anim = animation.FuncAnimation(fig, animate,
                                    frames = 500, 
                                    interval = 1000)
        # Once the Animation Function is ran, plt.show() is necesarry, 
        # otherwise it won't show the image. Also, plt.show() will stop the execution of the code 
        # that is located after. So if we want to continue with the following code, we must close the 
        # tab generated by matplotlib.   
        plt.show()


        # The next lines allow us to create a CSV file with data retrieved from E4 wristband.
        # This code is repeated if a KeyboardInterrupt exception arises as a redundant case
        # for storing the data recorded.
        with open("{}/Raw/fileBVP.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueBVP'])
            writer.writerows(BVP_array)

        with open("{}/Raw/fileACC.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueACC'])
            writer.writerows(Acc_array)   

        with open("{}/Raw/fileEDA.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueEDA'])
            writer.writerows(GSR_array)        

        with open("{}/Raw/fileTemp.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueTemp'])
            writer.writerows(Temp_array)        

        with open("{}/Raw/fileIBI.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueIBI'])
            writer.writerows(IBI_array)   

        # These next instructions should be executed only once, and exactly where we want the program to finish.
        # Otherwise, it may rise a Socket Error. These lines also written below in case of a KeyBoardInterrupt 
        # exception arising.
        global s
        print("Disconnecting from device")
        s.send("device_disconnect\r\n".encode())
        s.close()

    except KeyboardInterrupt:
        print('hola')
        #The next lines allow us to create a CSV file with data retrieved from E4 wristband.
        with open("{}/Raw/fileBVP.csv".format(folder), 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'valueBVP'])
            writer.writerows(BVP_array)

        with open("{}/Raw/fileACC.csv".format(folder), 'w', newline = '') as document:
                    writer = csv.writer(document)
                    writer.writerow(['Datetime', 'valueACC'])
                    writer.writerows(Acc_array)   

        with open("{}/Raw/fileEDA.csv".format(folder), 'w', newline = '') as document:
                    writer = csv.writer(document)
                    writer.writerow(['Datetime', 'valueEDA'])
                    writer.writerows(GSR_array)        

        with open("{}/Raw/fileTemp.csv".format(folder), 'w', newline = '') as document:
                    writer = csv.writer(document)
                    writer.writerow(['Datetime', 'valueTemp'])
                    writer.writerows(Temp_array)        

        with open("{}/Raw/fileIBI.csv".format(folder), 'w', newline = '') as document:
                    writer = csv.writer(document)
                    writer.writerow(['Datetime', 'valueIBI'])
                    writer.writerows(IBI_array)   

        # Once we have the data stored locally on CSV files, we store it in json files to send them through a socket.
        csvFilePath = 'fileBVP.csv'
        # jsonPost(csvFilePath, 'valueBVP')
        csvFilePath = 'fileACC.csv'
        # jsonPost(csvFilePath, 'valueACC')
        csvFilePath = 'fileEDA.csv'
        # jsonPost(csvFilePath, 'valueEDA')
        csvFilePath = 'fileTemp.csv'
        # jsonPost(csvFilePath, 'valueTemp')
        csvFilePath = 'fileIBI.csv'
        # jsonPost(csvFilePath, 'valueIBI')

        # We close connections
        print("Disconnecting from device")
        s.send("device_disconnect\r\n".encode())
        s.close()

# # CODE FOR EEG # #
def EEG(second, folder):
    # PSD 128 rows
    # Band power 1 row each

    ############### Alpha/Beta plot created ###############
    # Create the grid plot considering 4 channels.
    fig3 = plt.figure(3, constrained_layout=True, figsize=(10,6))
    gs3 = fig3.add_gridspec(2, 2)  # (Rows, Columns)
    ax17_20 = gs3.subplots(sharex=True, sharey=False)

    # The ranges of spectral signals are declared, though can be modified.
    spectral_signals = {'Alpha': [8, 12], 'Beta': [12, 30],
                        'Gamma': [30, 50], 'Theta': [4, 8], 'Delta': [1, 4]}

    # The following object will save parameters to connect with the EEG.
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    # MAC Adress is the only required parameters for ENOPHONEs
    params.mac_address = 'f4:0e:11:75:75:ce'
    # params.serial_port = 'COM4'

    # Relevant board IDs available:
    board_id = BoardIds.ENOPHONE_BOARD.value # (37)
    # board_id = BoardIds.SYNTHETIC_BOARD.value # (-1)
    # board_id = BoardIds.CYTON_BOARD.value # (0)

    # Relevant variables are obtained from the current EEG.
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eta = 4 # Order number of filter
    ft = 0 # Filter type (0: Butter, 1: Chev, 2: Bessel)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    board = BoardShim(board_id, params)

    # An empty dataframe is created to save Alpha/Beta values to plot in real time.
    alpha_beta_data = pd.DataFrame(columns=['Alpha_C' + str(c) for c in range(1, len(eeg_channels) + 1)])
    ####################################################################

    ############# Session is then initialized #######################
    board.prepare_session()
    # board.start_stream () # use this for default options
    board.start_stream(45000, "file://{}/testOpenBCI.csv:w".format(folder))
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Starting the streaming with Cyton ---')

    try:
        while (True):
            time.sleep(2)
            with second.get_lock():
                # When the seconds reach 332, we exit the functions.
                if(second.value == tiempo_prueba):
                    plt.close()
                    return
            data = board.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.
            eeg_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

            ############## Data collection #################

            # A list of combinations between channels and spectral signals is created.
            columns_signals = [s + '_' + 'C' + str(c) for c in range(1, len(eeg_channels) + 1) for s in
                               spectral_signals.keys()]
            l_signals = []

            # Empty DataFrames are created for raw and PSD data.
            df_crudas = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            df_psd = pd.DataFrame(columns=['PSD' + str(channel) for channel in range(1, len(eeg_channels) + 1)])

            # The total number of EEG channels is looped to obtain MV and PSD for each channel, and
            # thus saved it on the corresponding columns of the respective DataFrame.
            for eeg_channel in eeg_channels:
                data_channel = data[eeg_channel]
                
                df_crudas['MV' + str(eeg_channel)] = data_channel

                # 60 Hz Notch filter
                DataFilter.perform_bandstop(data=data_channel, sampling_rate=sampling_rate, center_freq=60,
                                            band_width=1, order=eta, filter_type=ft, ripple=0)

                # 0.1-100 Hz 4th order Butterworth bandpass filter
                DataFilter.perform_lowpass(data=data_channel, sampling_rate=sampling_rate, cutoff=50,
                                            order=eta, filter_type=ft, ripple=0)
                DataFilter.perform_highpass(data=data_channel, sampling_rate=sampling_rate, cutoff=1,
                                            order=eta, filter_type=ft, ripple=0)

                # Linear de-trend
                DataFilter.detrend(data_channel, DetrendOperations.LINEAR.value)

                psd_data = DataFilter.get_psd_welch(data_channel, nfft, nfft // 2, sampling_rate,
                                                    WindowFunctions.BLACKMAN_HARRIS.value)
                df_psd['PSD' + str(eeg_channel)] = psd_data[0]

                # Afterwards, a for loop goes over all the spectral signals and saves the unique
                # values into a list, posible because only 1 value is generated per second.
                for spectral_signal in spectral_signals.keys():
                    l_signals.append(DataFilter.get_band_power(psd_data, spectral_signals[spectral_signal][0],
                                                               spectral_signals[spectral_signal][1]))

            # The DataFrame that involves the spectral signals is created, using the previously created
            # list of combinations and our appended values when looping over each channel and signal.
            data_signals = pd.DataFrame(dict(zip(columns_signals, l_signals)), index=[eeg_time], columns=columns_signals)

            def get_columns_signal(spect_signal):
                """
                This function returns a list of combinations between channels and a given spectral signal.

                :param string spect_signal: Name of the spectral signal (Alpha, Beta, ...)
                :return list: List of combinations between channels and the spectral signal.
                """
                return [spect_signal + '_' + 'C' + str(c) for c in range(1, len(eeg_channels) + 1)]

            # Combined signals (Alpha / Beta) are created using all alphas and all betas from all given EEG channels.
            for eeg_channel in eeg_channels:
                data_signals['Alpha_Beta_C' + str(eeg_channel)] = data_signals['Alpha_C' + str(eeg_channel)] / data_signals['Beta_C' + str(eeg_channel)]

            # For each spectral signal, a CSV is saved according to its name, looping over the DataFrame created.
            for spectral_signal in list(spectral_signals.keys()) + ['Alpha_Beta']:
                data_signals.loc[:, get_columns_signal(spectral_signal)].to_csv('{}/Raw/{}.csv'.format(folder, spectral_signal), mode='a')

            # Both the raw and PSD DataFrame is exported as a CSV.
            df_crudas.to_csv('{}/Raw/Crudas.csv'.format(folder), mode='a')
            df_psd.to_csv('{}/Raw/PSD.csv'.format(folder), mode='a')
            df_psd.to_json('{}/Raw/PSD.json'.format(folder))

            ###########################################################################################################

            # # # # Alpha / Beta Plot usage # # # 

            # The empty Alpha/Beta DataFrame is appended with the current value, this would be
            # the only value that could be saved for each iteration, all other are overwritten.
            
            alpha_beta_data = alpha_beta_data.append(data_signals.loc[:, get_columns_signal('Alpha')], ignore_index=True)

            def plot_signal(ax, canal, n):
                """
                This function takes an axes object and plots the Alpha/Beta array according to the channel n.
                :param matplotlib.axes._subplots.AxesSubplot ax: Axes where the band power will be plotted. 
                :param pd.Series canal: The current column that is being plotted.
                :param integer n: Number of the current channel.
                """

                # The previous plot is cleared, new data is added and the title is re-established.
                ax.clear()
                ax.plot(canal)
                ax.set_title('Alpha / Beta Channel ' + str(n))
                plt.pause(0.001)

            # The following nested for loops go over the matrix of plots, and uses the previously declared
            # function to plot all the Alpha/Beta data for each channel on their respective plot.
            
            for i in range(ax17_20.shape[0]):
               for j in range(ax17_20.shape[1]):
                   if (i + j) != 4:
                    plot_signal(ax17_20[i][j], alpha_beta_data.iloc[:, i*2 + j], i*2 + j + 1)

    except KeyboardInterrupt:
        board.stop_stream()
        board.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- End the session with Cyton ---')

    ##############Links que pueden ayudar al entendimiento del código ##############
    # https://www.geeksforgeeks.org/python-save-list-to-csv/

# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.
if __name__ == '__main__':

    # # Define the data folder # #
    # The name of the folder is defined depending on the user's input
    subject_ID, repetition_num = input('Please enter the subject ID and the number of repetition: ').split(' ')
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    repetition_num = '0' + repetition_num if int(repetition_num) < 10 else repetition_num
    folder = 'S{}R{}_{}'.format(subject_ID, repetition_num, datetime.now().strftime("%d%m%Y_%H%M"))
    os.mkdir(folder)

    for subfolder in ['Raw', 'Processed', 'Figures']:
        os.mkdir('{}/{}'.format(folder, subfolder))

    # # Create a multiprocessing List # # 
    # This list will store the seconds where a beep was played
    timestamps = multiprocessing.Manager().list()
    
    # # Start processes # #
    process2 = Process(target=timer, args=[seconds, counts, timestamps])
    p = Process(target=empatica, args=[seconds, folder]) # Descomentar para Empatica
    q = Process(target=EEG, args=[seconds, folder])
    process2.start()
    p.start() # Descomentar para Empatica
    q.start()
    process2.join()
    p.join() # Descomentar para Empatica
    q.join()

    # # DATA STORAGE SECTION # #
    # Executed only once the test has finished.
    print(Fore.RED + 'Test finished sucessfully, storing data now...' + Style.RESET_ALL)
    # # Save beeps' timestamps in a .csv file # #
    # We must first convert the multiprocess.Manger.List to a normal list
    timestamps_final = list(timestamps)

    # Now we convert each of the UNIX-type timestamps to normal timestam (year-month-day hour-minute-second-ms)
    for i in range(len(timestamps_final)):
        timestamp = datetime.fromtimestamp(timestamps_final[i])
        timestamps_final[i] = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Store data in a .csv
    timestamps_final = pd.Series(timestamps_final)   
    df = pd.DataFrame(timestamps_final, columns=['Beep_Timestamp'])
    df.to_csv('{}/Timestamps.csv'.format(folder), index=False)
    print(Fore.GREEN + 'Data stored sucessfully' + Style.RESET_ALL)
    
    # # Data processing # #
    print(Fore.RED + 'Data being processed...' + Style.RESET_ALL)


    def remove_outliers(df, method):
        """
        Uses an statistical method to remove outlier rows from the dataset x, and filters the valid rows back to y.

        :param pd.DataFrame df: with non-normalized, source variables.
        :param string method: type of statistical method used.
        :return pd.DataFrame: Filtered DataFrame
        """

        # The number of initial rows is saved.
        n_pre = df.shape[0]

        # A switch case selects an statistical method to remove rows considered as outliers.
        if method == 'z-score':
            z = np.abs(stats.zscore(df))
            df = df[(z < 3).all(axis=1)]
        elif method == 'quantile':
            q1 = df.quantile(q=.25)
            q3 = df.quantile(q=.75)
            iqr = df.apply(stats.iqr)
            df = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
        
        # The difference between the processed and raw rows is printed.
        n_pos = df.shape[0]
        diff = n_pre - n_pos
        print(f'{diff} rows removed {round(diff / n_pre * 100, 2)}%')
        # print('{} rows removed {}%'.format(diff, round(diff / n_pre * 100, 2)))
        # print(str(diff) + 'rows removed' + str(round(diff / n_pre * 100, 2)) + '%')
        return df
    
    # The following for loop iterates over all features, and removes outliers depending on the statistical method used.
    # It reads the files saved in the "Raw" folder, and only reads .CSV files, to outputt a .CSV file in "Processed" folder.
    for df_name in os.listdir('{}/Raw/'.format(folder)):
        if df_name[-4:] == '.csv' and df_name[:4] != 'file':
            df_name = df_name[:-4]
            df_raw = pd.read_csv('{}/Raw/{}.csv'.format(folder, df_name), index_col=0)
            df_processed = remove_outliers(df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True), 'quantile')
            
            # The processed DataFrame is then exported to the "Processed" folder, and plotted.
            df_processed.to_csv('{}/Processed/{}_processed.csv'.format(folder, df_name))
            df_processed.plot()
            
            # The plot of the processed DataFrame is saved in the "Figures" folder.
            plt.savefig('{}/Figures/{}_plot.png'.format(folder, df_name))

    print(Fore.GREEN + 'Data processed successfully' + Style.RESET_ALL)

####### Sources ########
# https://www.empatica.com/connect/session_view.php?id=1699109

# To understand Value data type and lock method read the following link:
# https://www.kite.com/python/docs/multiprocessing.Value  
# For suffle of array, check the next link and user "mdml" answer:
# https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones