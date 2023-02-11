import streamlit as st
from  PIL import Image
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px 
import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from streamlit_lottie import st_lottie
import json
import plotly.express as px
import pickle

img = Image.open('img.png')

def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)


st.set_page_config(page_title="Agri App",page_icon=img, layout="wide")

video_html = """
		<style>

		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		}

		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.5);
		  color: #f1f1f1;
		  width: 100%;
		  padding: 20px;
		}

		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="")>
		  Your browser does not support HTML5 video.
		</video>
        """

st.markdown(video_html, unsafe_allow_html=True)

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            body {
            background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
            background-size: cover;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv("crop_prediction_model_one.csv")

converts_dict = {
    'Nitrogen': 'N',
    'Phosphorus': 'P',
    'Potassium': 'K',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Rainfall': 'rainfall',
    'ph': 'ph'
}

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    input = np.array([[n, p, k, temperature, humidity, ph, rainfall]]).astype(np.float64)
    prediction = model.predict(input)
    return prediction[0]

def scatterPlotDrawer(x,y):
    fig = px.scatter(df, x=x, y=y,color='label')  # palette="deep",, sizes=(20, 200), legend="full"
    st.write(fig)

def barPlotDrawer(x,y):
    fig = px.bar(df, x=x, y=y)
    st.write(fig)

def boxPlotDrawer(x,y):
    fig = px.box(df,x=x, y=y)
    st.write(fig)

def main():
    html_temp_vis = """
    <div style="background-color:#025246 ;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Visualize Soil Properties </h2>
    </div>
    """

    html_temp_pred = """
    <div style="background-color:#025246 ;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Which Crop To Cultivate? </h2>
    </div>
    """


local_css("style.css")
global data
global numeric_columns
global non_numeric_columns
data = pd.read_csv("Complete.csv")

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["About","Visualization","Prediction"],
        icons=["house","clipboard-data","graph-up-arrow"],
        menu_icon=None,
        
        styles={
                "container": {"padding": "0!important", "background-color": "FFCCD2"},  #background color
                "icon": {"color": "black", "font-size": "20px"},   # icon color
                "nav-link": {
                    "font-size": "20px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#A7727D", # icon color after hover
                },
                "nav-link-selected": {"background-color": "#A7727D"},   # Colour when tab is selected  #ADD8E6
                "icon": {"color": "black", "font-size": "20px"},   # icon color
            },
        )

if selected=="About":
    st.title('*Agriculture*')
    with st.container():
        left_column, right_column = st.columns(2)
        with left_column:
                st.header("Crop Cultivation")
                st.subheader('*Introduction*')
                st.write("Agriculture is the art and science of cultivating the soil, growing crops and raising livestock. It includes the preparation of plant and animal products for people to use and their distribution to markets. Agriculture provides most of the world’s food and fabrics. Cotton, wool, and leather are all agricultural products. Agriculture also provides wood for construction and paper products. These products, as well as the agricultural methods used, may vary from one part of the world to another.")
        with right_column:
            lottie_Cancer = load_lottiefile("LottieFiles/Crop_1.json")
            st_lottie(
                lottie_Cancer,
                height=400,
                width=400,
            )
        with right_column:
            st.subheader('Crop Diseases')
            st.write("Generally, a plant gets diseased when it is continually disrupted by a certain causal agent, resulting in a physiological process anomaly that disrupts the normal structure of the plant’s function, and growth, among other activities. Pathological conditions and symptoms result from the disruption of one or more of a plant’s critical biochemical and physiological systems. The occurrence and prevalence of crop diseases vary seasonally, depending on the prevalence of a pathogen, conditions of the environment, and the crops and varieties grown. Some plant varieties are more prone to outbreaks of plant diseases than others.")
        with left_column:
            lottie_Cancer = load_lottiefile("LottieFiles/Crop_2.json")
            st_lottie(
                lottie_Cancer,
                height=400,
                width=400,
            )

        st.write("Based on various meteorological factors, climate changes, soil type and various other factors, the type of crop to be cultivated on a redion is decided")
        st.write("The following video shows how machine learning is being used in the domain of agriculture.")
        DEFAULT_WIDTH = 50
        VIDEO_DATA = "https://youtu.be/NdDrk2-7rRE"

        width = st.sidebar.slider(
            label="Width", min_value=0, max_value=100, value=DEFAULT_WIDTH, format="%d%%"
        )

        width = max(width, 0.01)
        side = max((100 - width) / 2, 0.01)

        _, container, _ = st.columns([side, width, side])
        container.video(data=VIDEO_DATA)


if selected == "Visualization":
        st.header("Visualization")
        st.write("Now its time for us to Visualize 📊 the dataset that we have uploaded. When speaking about visualization, We should als understand there are many types. Some of which are as follows: \n  1) Bar chart \n 2) Scatter Plot \n 3) Box plot \n")
        st.write("It is used to represent the data in a graphical manner📈 for better understanding.")
        st.write("In the Page click the visualize  to visualize the dataset that we have uploaded.")
        plot_type = st.selectbox("Select plot type", ('Bar Plot', 'Scatter Plot', 'Box Plot'))
        st.subheader("Relation between features")

        # Plot!

        x = ""
        y = ""

        if plot_type == 'Bar Plot':
            x = 'label'
            y = st.selectbox("Select a feature to compare between crops",
                ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
        if plot_type == 'Scatter Plot':
            x = st.selectbox("Select a property for 'X' axis",
                ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
            y = st.selectbox("Select a property for 'Y' axis",
                ('Nitrogen', 'Phosphorus', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
        if plot_type == 'Box Plot':
            x = "label"
            y = st.selectbox("Select a feature",
                ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))

        if st.button("Visulaize"):
            if plot_type == 'Bar Plot':
                y = converts_dict[y]
                barPlotDrawer(x, y)
            if plot_type == 'Scatter Plot':
                x = converts_dict[x]
                y = converts_dict[y]
                scatterPlotDrawer(x, y)
            if plot_type == 'Box Plot':
                y = converts_dict[y]
                boxPlotDrawer(x, y)

if selected == "Prediction":
        st.header("To predict your crop give values")

        st.write("When taking consideration some other parameters must also br taken into account for bringing the suitable conclusion.It's not always necessary that all the data will be given for carrying out the analysis. In such cases relying on the given data won't be sufficient enough.")
        st.write("So, We must use Machine Learning concepts instead to make accurate and absolute decisions about the data we are handling.")
        st.header("Approch toward researching the solution.")
        st.write("One of the most popular algorithm that is used for doing the job is Random Forest in which it used to predit the output")
        st.write("In the Page drag the values and click the predict to predict the dataset that we have uploaded.")

        state_select = st.selectbox(
        label="Select the State",
        options=['None', 'Andaman and Nicobar', 'Arunachal Pradesh', 'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'])
        if state_select == 'Andaman and Nicobar':
                    x = st.selectbox('Select the District', options=['None', 'South Andaman','North & Middle Andaman','Nicobars'])
        if state_select == 'Arunachal Pradesh':
                    y = st.selectbox('Select the District', options=['None', 'Anjaw','Capital Complex Itanagar','Changlang', 'Dibang Valley', 'East Kameng', 'East Siang', 'Kamle', 'Kra Daadi', 'Kurung Kumey', 'Lepa Rada', 'Lohit', 'Longding', 'Lower Siang', 'Lower Subansiri', 'Namsai', 'Pakke Kessang','Papum Pare', 'SHI YOMI', 'Siang', 'Tawang', 'Tirap', 'Upper Siang', 'Upper Subansiri', 'West Kameng', 'West Siang'])        
        if state_select == 'Assam':
                    a = st.selectbox('Select the District', options=['None','Baksa','Barpeta','Biswanath', 'Bongaigaon', 'Cachar', 'Charaideo', 'Chirang', 'Darrang', 'Dhemaji', 'Dhubri', '	Dibrugarh', 'Dima Hasao', 'Goalpara', 'Golaghat', 'Hailakandi', 'Hojai', 'Jorhat', 'Kamrup Metropolitan', 'Kamrup', 'Karbi Anglong', 'Karimganj', 'Kokrajhar', 'Lakhimpur', 'Majuli', 'Morigaon', 'Nagaon', 'Nalbari', 'Sivasagar', 'Sonitpur', 'South Salmara-Mankachar', 'Tinsukia', 'Udalguri', 'West Karbi Anglong'])
        if state_select == 'Bihar':
                    b = st.selectbox('Select the District', options=['None','Araria', 'Arwal', 'Aurangabad', 'Banka', 'Begusarai', 'Bhagalpur', 'Bhojpur', 'Buxar', '	Darbhanga', 'East Champaran', 'Gaya', 'Gopalganj', 'Jamui', 'Jehanabad', 'Khagaria', 'Kishanganj', 'Kaimur', 'Katihar', 'Lakhisarai', 'Madhubani', 'Munger', 'Madhepura', 'Muzaffarpur', 'Nalanda', 'Nawada', 'Patna', 'Purnia', 'Rohtas', 'Saharsa', 'Samastipur', 'Sheohar', 'Sheikhpura', 'Saran', 'Sitamarhi', 'Supaul', 'Siwan', 'Vaishali', 'West Champaran'])
        if state_select == 'Chattishgarh':
                    c = st.selectbox('Select the District', options=['None','Balod', 'Baloda Bazar', 'Balrampur', 'Bastar', 'Bemetara', 'Bijapur', 'Bilaspur', 'Dantewada', 'Dhamtari', 'Durg', 'Gariaband', 'Janjgir-Champa', 'Jashpur', 'Kabirdham','Kanker', 'Kondagaon', 'Korba', 'Koriya', 'Mahasamund', 'Mungeli', 'Narayanpur', 'Raigarh', 'Raipur', 'Rajnandgaon', 'Sukma', 'Surajpur', 'Surguja'])
        if state_select == 'Goa':
                    d = st.selectbox('Select the District', options=['None','North Goa', 'South Goa'])
        if state_select == 'Gujarat':
                    e = st.selectbox('Select the District', options=['None','Ahmedabad', 'Amreli', 'Anand', 'Aravalli', 'Banaskantha', 'Bharuch', 'Bhavnagar', 'Botad', 'Chhota Udaipur', 'Dahod', 'Dang', 'Devbhoomi Dwarka', 'Gandhinagar', 'Gir Somnath', 'Jamnagar', 'Junagadh', 'Kutch', 'Kheda', 'Mahisagar', 'Mehsana', 'Morbi', 'Narmada', 'Navsari', 'Panchmahal', 'Patan', 'Porbandar', 'Rajkot', 'Sabarkantha', 'Surat', 'Surendranagar', 'Tapi', 'Vadodara', 'Valsad'])
        if state_select == 'Haryana':
                    f = st.selectbox('Select the District', options=['None','Ambala','Bhiwani', 'Charkhi Dadri', 'Faridabad','Fatehabad','Gurugram','Hisar','Jhajjar','Jind', 'Kaithal', 'Karnal', 'Kurukshetra', 'Mahendragarh', 'Nuh', 'Palwal', 'Panchkula', 'Panipat', 'Rewari', 'Rohtak', 'Sirsa', 'Sonipat', 'Yamunanagar'])
        if state_select == 'Himachal Pradesh':
                    h = st.selectbox('Select the District', options=['None','Bilaspur', 'Chamba','Hamirpur', 'Kangra', 'Kinnaur', 'Kullu', 'Lahaul and Spiti', 'Mandi', 'Shimla', 'Sirmaur', 'Solan', 'Una'])
        if state_select == 'Jharkhand':
                    i = st.selectbox('Select the District', options=['None','Bokaro', 'Chatra', 'Deoghar', 'Dhanbad', 'Dumka', 'East Singhbhum', 'Garhwa', 'Giridih', 'Godda', 'Gumla', 'Hazaribagh', 'Jamtara', 'Khunti', 'Koderma', 'Latehar', 'Lohardaga', 'Pakur', 'Palamu', 'Ramgarh', 'Ranchi', 'Sahebganj', 'Seraikela', 'Simdega', 'West Singhbhum'])
        if state_select == 'Karnataka':
                    j = st.selectbox('Select the District', options=['None','Bagalkot', 'Bengaluru Urban', 'Bengaluru Rural', 'Belagavi', 'Ballari', 'Bidar', 'Vijayapur', 'Chamarajanagar', 'Chikballapur', 'Chikkamagaluru', 'Chitradurga', 'Dakshina Kannada', 'Davanagere', 'Dharwad', 'Gadag', 'Kalaburagi', 'Hassan', 'Haveri', 'Kodagu', 'Kolar', 'Koppal', 'Mandya', 'Mysuru', 'Raichur', 'Ramanagara', 'Shivamogga', 'Tumakuru', 'Udupi', 'Uttara Kannada', 'Yadgir'])
        if state_select == 'Madhya Pradesh':
                    k = st.selectbox('Select the District', options=['None','Agar Malwa', 'Alirajpur', 'Anuppur', 'Ashoknagar', 'Balaghat', 'Barwan', 'Betul', 'Bhind', 'Bhopal', 'Burhanpur', 'Chhatarpur', 'Chhindwara', 'Damoh', 'Datia', 'Dewas', 'Dhar', 'Dindori', 'East Nimar', 'Guna', 'Gwalior', 'Harda', 'Hoshangabad', 'Indore', 'Jabalpur', 'Jhabua', 'Katni', 'Mandla', 'Mandsaur', 'Morena', 'Narsinghpur', 'Neemuch', 'Niwari', 'Panna', 'Raisen', 'Rajgarh', 'Ratlam', 'Rewa', 'Sagar', 'Satna', 'Sehore', 'Seoni', 'Shahdol', 'Shajapur', 'Sheopur', 'Shivpuri', 'Sidhi', 'Singrauli', 'Tikamgarh', 'Ujjain', 'Umaria', 'Vidisha', 'West Nimar'])
        if state_select == 'Maharashtra':
                    l = st.selectbox('Select the District', options=['None','Ahmednagar', 'Akola', 'Amravati', 'Aurangabad', 'Beed', 'Bhandara', 'Buldhana', 'Chandrapur', 'Dhule', 'Gadchiroli', 'Gondia', 'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur', 'Latur', 'Mumbai City', 'Mumbai Suburban', 'Nagpur', 'Nanded', 'Nandurbar', 'Nashik', 'Osmanabad', 'Palghar', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara', 'Sindhudurg', 'Solapur', 'Thane', 'Wardha', 'Washim', 'Yavatmal'])
        if state_select == 'Manipur':
                    m = st.selectbox('Select the District', options=['None','Bishnupur', 'Thoubal', 'Imphal East', 'Imphal West', 'Senapati', 'Ukhrul', 'Chandel', 'Churachandpur', 'Tamenglong'])
        if state_select == 'Meghalaya':
                    n = st.selectbox('Select the District', options=['None','East Garo Hills', 'East Khasi Hills', 'Jaintia Hills', 'Ri Bhoi', 'South Garo Hills', 'West Garo Hills', 'West Khasi Hills'])
        if state_select == 'Mizoram':
                    o = st.selectbox('Select the District', options=['None','Aizawl', 'Kolasib', 'Lawngtlai', 'Lunglei', 'Mamit', 'Saiha', 'Serchhip', 'Champhai'])
        if state_select == 'Nagaland':
                    p = st.selectbox('Select the District', options=['None','Dimapur', 'Kiphire', 'Kohima', 'Longleng', 'Mokokchung', 'Mon', 'Peren', 'Phek', 'Tuensang', 'Wokha', 'Zunheboto', 'Noklak'])
        if state_select == 'Odisha':
                    q = st.selectbox('Select the District', options=['None','Angul', 'Boudh (Baudh)', 'Balangir', 'Bargarh', 'Balasore (Baleswar)', 'Bhadrak', 'Cuttack', 'Deogarh (Debagarh)', 'Dhenkanal', 'Ganjam', 'Gajapati', 'Jharsuguda', 'Jajpur', 'Jagatsinghapur', 'Khordha','Keonjhar (Kendujhar)', 'Kalahandi', 'Kandhamal', 'Koraput', 'Kendrapara', 'Malkangiri', 'Mayurbhanj', 'Nabarangpur', 'Nuapada', 'Nayagarh', 'Puri', 'Rayagada', 'Sambalpur', 'Subarnapur (Sonepur)', 'Sundargarh'])  
        if state_select == 'Punjab':
                    r = st.selectbox('Select the District', options=['None','Amritsar', 'Barnala', 'Bathinda', 'Faridkot', 'Fatehgarh Sahib', 'Firozpur', 'Fazilka', 'Gurdaspur', 'Hoshiarpur', 'Jalandhar', 'Kapurthala', 'Ludhiana', 'Mansa', 'Moga','Sri Muktsar Sahib', 'Pathankot', 'Patiala', 'Rupnagar', 'Sahibzada Ajit Singh Nagar', 'Sangrur', 'Shahid Bhagat Singh Nagar', 'Taran Taran'])
        if state_select == 'Rajasthan':
                    s = st.selectbox('Select the District', options=['None','Ajmer', 'Alwar', 'Banswara', 'Baran', 'Barmer', 'Bharatpur', 'Bhilwara', 'Bikaner', 'Bundi', 'Chittorgarh', 'Churu', 'Dausa', 'Dholpur', 'Dungarpur', 'Hanumangarh', 'Jaipur', 'Jaisalmer', 'Jalor','Jhalawar', 'Jhunjhunu', 'Jodhpur', 'Karauli', 'Kota', 'Nagaur', 'Pali', 'Pratapgarh', 'Rajsamand', 'Sawai Madhopur', 'Sikar', 'Sirohi', 'Sri Ganganagar', 'Tonk', 'Udaipur'])   
        if state_select == 'Sikkim':
                    t = st.selectbox('Select the District', options=['None','East Sikkim', 'North Sikkim', 'South Sikkim', 'West Sikkim']) 
        if state_select == 'Tamil Nadu':
                    u = st.selectbox('Select the District', options=['None','Ariyalur', 'Chennai', 'Coimbatore', 'Cuddalore', 'Dharmapuri', 'Dindigul', 'Erode', 'Kallakurichi', 'Kanchipuram', 'Kanniyakumari', 'Karur', 'Krishnagiri', 'Madurai', 'Nagapattinam', 'Namakkal', 'Nilgiris', 'Perambalur', 'Pudukkottai', 'Ramanathapuram', 'Salem', 'Sivagangai', 'Thanjavur', 'Theni', 'Thoothukudi', 'Tiruchirappalli', 'Tirunelveli', 'Tiruppur', 'Tiruvallur', 'Tiruvannamalai', 'Tiruvarur', 'Vellore', 'Viluppuram', 'Virudhunagar'])    
        if state_select == 'Telengana':
                    v = st.selectbox('Select the District', options=['None','Adilabad', 'Bhadradri Kothagudem', 'Hyderabad', 'Jagitial', 'Jangaon', 'Jayashankar Bhupalapally', 'Jogulamba Gadwal', 'Kamareddy', 'Karimnagar', 'Khammam', 'Kumarambheem Asifabad', 'Mahabubabad', 'Mahabubnagar', 'Mancherial district', 'Medak', 'Medchal–Malkajgiri', 'Mulugu', 'Nagarkurnool', 'Narayanpet', 'Nalgonda', 'Nirmal', 'Nizamabad', 'Peddapalli', 'Rajanna Sircilla', 'Ranga Reddy', 'Sangareddy', 'Siddipet', 'Suryapet', 'Vikarabad', 'Wanaparthy', 'Warangal Rural', 'Warangal Urban', 'Yadadri Bhuvanagiri'])  
        if state_select == 'Tripura':
                    w = st.selectbox('Select the District', options=['None','Dhalai', 'North Tripura', 'South Tripura', 'West Tripura'])  
        if state_select == 'Uttar Pradesh':
                    z = st.selectbox('Select the District', options=['None','Agra', 'Aligarh', 'PrayagRaj', 'Ambedkar Nagar', 'Amroha', 'Auraiya', 'Azamgarh', 'Badaun', 'Bahraich', 'Ballia', 'Balrampur', 'Banda District', 'Barabanki', 'Bareilly', 'Basti', 'Bijnor', 'Bulandshahr', 'Chandauli(Varanasi Dehat)', 'Chitrakoot', 'Deoria', 'Etah', 'Etawah', 'Faizabad', 'Farrukhabad', 'Fatehpur', 'Firozabad', 'Gautam Buddha Nagar', 'Ghaziabad', 'Ghazipur', 'Gonda', 'Gorakhpur', 'Hamirpur', 'Hapur District', 'Hardoi', 'Hathras', 'Jaunpur District', 'Jhansi', 'Kannauj', 'Kanpur Dehat', 'Kanpur Nagar', 'Kasganj', 'Kaushambi', 'Kushinagar', 'Lakhimpur Kheri', 'Lalitpur', 'Lucknow', 'Maharajganj', 'Mahoba', 'Mainpuri', 'Mathura', 'Mau', 'Meerut', 'Mirzapur', 'Moradabad', 'Muzaffarnagar', 'Pilibhit', 'Pratapgarh', 'Rae Bareli', 'Rampur', 'Saharanpur', 'Sant Kabir Nagar', 'Sant Ravidas Nagar', 'Sambhal', 'Shahjahanpur', 'Shamli', 'Shravasti', 'Siddharthnagar', 'Sitapur', 'Sonbhadra', 'Sultanpur', 'Unnao', 'Varanasi (Kashi)', 'Allahabad', 'Amethi', 'Bagpat'])  
        if state_select == 'Uttarakhand':
                    ab = st.selectbox('Select the District', options=['None','Almora', 'Bageshwar', 'Chamoli', 'Champawat', 'Dehradun', 'Haridwar', 'Nainital', 'Pauri Garhwal', 'Pithoragarh', 'Rudraprayag', 'Tehri Garhwal', 'Udham Singh Nagar', 'Uttarkashi'])  
        if state_select == 'West Bengal':
                    bc = st.selectbox('Select the District', options=['None','Alipurduar', 'Bankura', 'Birbhum', 'Cooch Behar', 'Dakshin Dinajpur', 'Darjeeling', 'Hooghly', 'Howrah', 'Jalpaiguri', 'Jhargram', 'Kalimpong', 'Kolkata', 'Malda', 'Murshidabad', 'Nadia', 'North 24 Parganas', 'Paschim Bardhaman', 'Paschim Medinipur', 'Purba Bardhaman', 'Purba Medinipur', 'Purulia', 'South 24 Parganas', 'Uttar Dinajpur'])  
        if state_select == 'Andhra Pradesh':
                    x = st.selectbox('Select the District', options=['None', 'Anantapur','Chittoor','East Godavari', 'Guntur', 'Kadapa', 'Krishna', 'Kurnool', 'Sri Potti Sri Ramulu Nellore', 'Prakasam', 'Srikakulam', 'Visakhapatnam', 'Vizianagaram', 'West Godavari'])
        season_select = st.selectbox(
            label= "Select the Season",
            options=['None', 'Whole Year', 'Autumn', 'Monsoon', 'Winter', 'Summer'])

        st.subheader("Drag to Give Values")
        n = st.slider('Nitrogen', 0, 140)
        p = st.slider('Phosphorus', 5, 145)
        k = st.slider('Potassium', 5, 205)
        temperature = st.slider('Temperature', 8.83, 43.68)
        humidity = st.slider('Humidity', 14.26, 99.98)
        ph = st.slider('pH', 3.50, 9.94)
        rainfall = st.slider('Rainfall', 20.21, 298.56)
        
        if st.button("Predict your crop"):
            output=predict_crop(n, p, k, temperature, humidity, ph, rainfall)
            res = "“"+ output.capitalize() + "”"
            st.success('The most suitable crop for your field is {}'.format(res))

if __name__=='__main__':
    main()
    
