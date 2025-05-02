# Turkish Cup Tournament Simulator 🏆⚽

This is a full-stack football tournament simulator that models a fictional **Turkish Cup** using historical Süper Lig data. It includes group and knockout stages, match predictions powered by a trained ML model, and an interactive frontend UI.

## 📦 Features

### 🖥️ Frontend (React + Firebase)
- Simulates full Turkish Cup tournaments
- Group stage and knockout bracket view
- Goal scorers and in-match timelines
- Team statistics and top scorers
- Fully responsive design (TailwindCSS)
- Hosted on Firebase

### 🧠 Backend (Flask + ML)
- Trained on Turkish Super League match data
- Predicts match scores using Random Forest regressors
- Exposes API endpoints for predictions and statistics
- Designed to run on AWS EC2 (or any server)

---

## 🔧 Technologies Used

| Layer     | Stack                                         |
|-----------|-----------------------------------------------|
| Frontend  | React, TailwindCSS, Recharts, Firebase Hosting |
| Backend   | Flask, Pandas, Scikit-learn, joblib, CORS     |
| ML Model  | RandomForestRegressor from scikit-learn       |
| Dataset   | Turkish Süper Lig historical match results    |

---

## 📁 Project Structure

RegessionModel/
│
├── backend/                # Flask API with ML model
│   ├── app.py              # Main Flask app
│   ├── tsl_dataset.csv     # Dataset used for training
│   └── models/             # Saved models and dataset
│
├── frontend/               # React frontend
│   ├── public/             # Static files
│   ├── src/                # React components and pages
│   └── firebase.json       # Firebase hosting config
│
├── requirements.txt        # Python dependencies
└── README.md               # You’re here!

---

## 🚀 Deployment

### ✅ Frontend (Firebase)
```bash
cd frontend
npm install
npm run build
firebase deploy

✅ Backend (EC2)

cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

Make sure EC2 allows inbound traffic to port 5000 or proxy it behind Nginx.

⸻

🔗 Live Demo

Frontend: https://your-firebase-url.web.app
Backend: Deployed separately on EC2 (not publicly linked)

⸻

📣 Credits
	•	Turkish Süper Lig dataset by faruky on Kaggle
	•	Built by Berke Aktürk as a full-stack sports analytics project.

⸻

📬 Contact

If you have questions or suggestions, feel free to reach out at:
📧 contact [at] berkeakturk.com 
🌐 github.com/berkeakturk1
