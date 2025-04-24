# Fraud Detection Blockchain System

This project integrates a machine learning-based fraud detection system with blockchain for secure and transparent prediction recording.

## Prerequisites

- Python 3.8+
- Node.js and npm
- Ganache (local Ethereum blockchain)
- MetaMask browser extension

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn web3
```

### 2. Start Ganache

Launch Ganache with the following settings:
- Port: 7545
- Network ID: 1337
- Gas Limit: 6721975

### 3. Deploy Smart Contract

```bash
# Compile contracts
npm run compile

# Deploy to Ganache
npm run deploy
```

### 4. Connect MetaMask to Ganache

- Open MetaMask
- Add a new network:
  - Network Name: Ganache
  - RPC URL: http://127.0.0.1:7545
  - Chain ID: 1337
  - Currency Symbol: ETH
- Import an account from Ganache using the private key

### 5. Start the Fraud Detection Application

```bash
npm run start
```

## Usage

1. The application will open in your browser
2. Select an Ethereum account from the sidebar
3. Enter transaction details
4. Click "Predict Fraud and Record on Blockchain"
5. The prediction will be recorded on the blockchain
6. You can view recent predictions in the blockchain explorer

## Project Structure

- `contracts/`: Smart contract code
- `scripts/`: Deployment scripts
- `frontend/`: Web interface files
- `Prediction.py`: Main application with Streamlit UI

## Note on Dataset

The application expects a fraud detection dataset. For testing, you can use:
- A sample credit card fraud dataset
- Make sure to update the file path in `Prediction.py`