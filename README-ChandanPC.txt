Credit Card Fraud Detection with Blockchain Integration
=================================================================

This is a full-stack project that combines machine learning and blockchain to detect and record fraudulent credit card transactions. It uses Streamlit for the UI, Hardhat for Ethereum smart contract deployment, and Ganache or MetaMask for blockchain interaction.

🚀 Features
-----------
- ✅ Real-time fraud prediction using ML models (Logistic Regression, Random Forest, XGBoost)
- 🔁 Class balancing using SMOTE
- 🔒 Blockchain-backed transaction logging using Ethereum smart contracts
- 🌐 Web frontend using Web3.js for blockchain interaction
- 📊 Model performance dashboard

📁 Project Structure
--------------------
FraudDetectionSystem/
├── app.py                     # Streamlit home UI
├── Prediction.py              # Streamlit prediction UI
├── performance.py             # Streamlit performance UI
├── models/                    # Stores trained models (.pkl)
├── sample_dataset.csv         # Sample dataset
├── frontend/
│   ├── index.html             # Web3 frontend dashboard
│   ├── app.js                 # Web3 JavaScript logic
│   └── src/
│       └── artifacts/         # Hardhat ABI and build files
├── contracts/
│   ├── FraudDetection.sol     # Main smart contract
│   └── ModelVerification.sol  # (optional) Additional contract
├── scripts/
│   └── deploy.js              # Contract deployment script
├── hardhat.config.js          # Hardhat config
├── package.json               # Node dependencies
├── requirements.txt           # Python dependencies
└── README.txt

🛠️ Prerequisites
-----------------
- Node.js and npm
- Python 3.8+
- Git
- MetaMask (browser extension)
- Ganache (or local Ethereum node)

🐍 Python Setup
---------------
1. Clone the repository:
   $ git clone https://github.com/yourusername/fraud-detection-blockchain.git
   $ cd fraud-detection-blockchain

2. Create and activate a virtual environment:
   $ python -m venv .venv
   $ source .venv/bin/activate     (on Windows: .venv\Scripts\activate)

3. Install dependencies:
   $ pip install -r requirements.txt

4. Start Streamlit app:
   $ streamlit run app.py

⚙️ Node & Hardhat Setup
-----------------------
1. Install Node dependencies:
   $ npm install

2. Compile the contract:
   $ npx hardhat compile

3. Deploy the smart contract:
   $ npx hardhat run scripts/deploy.js --network ganache

🌐 Launch Web Frontend
----------------------
1. Open frontend/index.html in your browser.
2. Connect MetaMask to the Ganache network.

🧪 Testing Smart Contracts
--------------------------
$ npx hardhat test
