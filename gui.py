import tkinter as tk
from tkinter import filedialog, Label, Button
import sounddevice as sd
import soundfile as sf
import numpy as np
from python_speech_features import mfcc

# Initializing the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Gender Detector by Voice')
top.configure(background='#CDCDCD')

# Initializing the Labels
gender_label = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
voice_file_label = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
sign_image = Label(top)

# Function to record voice and predict gender
def record_and_predict_gender():
    global gender_label
    voice_features = record_voice()
    gender = predict_gender(voice_features)
    gender_label.configure(foreground="#011638", text=gender)

# Function to record voice
def record_voice(sr=16000, channels=1, duration=3, filename='pred_record.wav'):
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels).reshape(-1)
    sd.wait()

    # Save the recording
    sf.write(filename, recording, sr)

    return get_MFCC(sr, recording)

# Function to extract MFCC features
def get_MFCC(sr, audio):
    features = mfcc(audio, samplerate=sr, winlen=0.025, winstep=0.01, numcep=13, appendEnergy=False)
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)  # Standardize features

    return features

# Function to predict gender based on pitch and energy
def predict_gender(voice_features):
    pitch_mean = np.mean(voice_features[:, 0])
    energy_mean = np.mean(voice_features[:, 1])

    # Adjust the thresholds as needed
    gender = "Detected gender:Male" if pitch_mean < 150 and energy_mean > -20 else "Detected gender:Female"

    return gender

# Defining Show Detect button function
def show_detect_button():
    detect_button = Button(top, text="Detect Gender", command=record_and_predict_gender, padx=10, pady=5)
    detect_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)

# Defining Upload Voice Function
def upload_voice():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
        voice_file_label.configure(text=file_path)
        show_detect_button()
    except:
        pass

upload = Button(top, text="Upload Voice File", command=upload_voice, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
voice_file_label.pack(side='bottom', expand=True)
sign_image.pack(side='bottom', expand=True)
gender_label.pack(side="bottom", expand=True)
heading = Label(top, text="Gender Detector by Voice", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()
top.mainloop()
