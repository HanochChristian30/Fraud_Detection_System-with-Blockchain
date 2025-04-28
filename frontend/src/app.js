// Web3 connection
let web3;
let contract;
let selectedAccount;
let currentFilter = 'all';

// DOM elements
const connectionStatus = document.getElementById('connection-status');
const accountsSelect = document.getElementById('accounts');
const accountBalance = document.getElementById('account-balance');
const contractStats = document.getElementById('contract-stats');
const recentPredictions = document.getElementById('recent-predictions');
const networkBadge = document.getElementById('network-badge');
const predictionCount = document.getElementById('prediction-count');
const fraudRate = document.getElementById('fraud-rate');

// Initialize the application
async function init() {
    try {
        // Connect to Web3
        if (window.ethereum) {
            web3 = new Web3(window.ethereum);
            try {
                // Request account access
                await window.ethereum.request({ method: 'eth_requestAccounts' });
                connectionStatus.innerHTML = '<div class="alert alert-success">Connected to MetaMask</div>';
                networkBadge.className = 'status-badge badge-connected';
                networkBadge.innerText = 'Connected';
            } catch (error) {
                connectionStatus.innerHTML = '<div class="alert alert-danger">MetaMask connection denied</div>';
                networkBadge.className = 'status-badge badge-disconnected';
                networkBadge.innerText = 'Disconnected';
            }
        } else if (window.web3) {
            web3 = new Web3(window.web3.currentProvider);
            connectionStatus.innerHTML = '<div class="alert alert-success">Connected to browser wallet</div>';
            networkBadge.className = 'status-badge badge-connected';
            networkBadge.innerText = 'Connected';
        } else {
            // Fallback to Ganache
            web3 = new Web3(new Web3.providers.HttpProvider('http://localhost:7545'));
            connectionStatus.innerHTML = '<div class="alert alert-success">Connected to Ganache</div>';
            networkBadge.className = 'status-badge badge-connected';
            networkBadge.innerText = 'Connected (Ganache)';
        }

        // Load the contract
        await loadContract();
        
        // Load accounts
        await loadAccounts();
        
        // Set up event listeners
        accountsSelect.addEventListener('change', accountChanged);
        
        // Load contract stats
        await loadContractStats();
        
        // Load recent predictions
        await loadRecentPredictions();
        
        // Listen for events
        listenForEvents();
        
    } catch (error) {
        console.error('Initialization error:', error);
        connectionStatus.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        networkBadge.className = 'status-badge badge-disconnected';
        networkBadge.innerText = 'Error';
    }
}

// Load the contract
async function loadContract() {
    try {
        // Load contract address
        const response = await fetch('./contractAddress.json');
        const contractData = await response.json();
        const contractAddress = contractData.FraudDetection;
        
        // Load contract ABI
        const abiResponse = await fetch('./artifacts/contracts/FraudDetection.sol/FraudDetection.json');
        const contractJson = await abiResponse.json();
        const contractAbi = contractJson.abi;
        
        // Initialize contract
        contract = new web3.eth.Contract(contractAbi, contractAddress);
        
        console.log('Contract loaded:', contract);
    } catch (error) {
        console.error('Contract loading error:', error);
        throw error;
    }
}

// Load accounts
async function loadAccounts() {
    try {
        const accounts = await web3.eth.getAccounts();
        
        if (accounts.length === 0) {
            accountsSelect.innerHTML = '<option value="">No accounts found</option>';
            return;
        }
        
        accountsSelect.innerHTML = '';
        accounts.forEach(account => {
            const option = document.createElement('option');
            option.value = account;
            option.textContent = `${account.substring(0, 8)}...${account.substring(account.length - 6)}`;
            accountsSelect.appendChild(option);
        });
        
        // Select first account
        selectedAccount = accounts[0];
        accountsSelect.value = selectedAccount;
        
        // Update balance
        updateAccountBalance();
    } catch (error) {
        console.error('Error loading accounts:', error);
        accountsSelect.innerHTML = '<option value="">Error loading accounts</option>';
    }
}

// Refresh accounts (for button)
function refreshAccounts() {
    loadAccounts();
}

// Update account balance
async function updateAccountBalance() {
    if (!selectedAccount) return;
    
    try {
        const balance = await web3.eth.getBalance(selectedAccount);
        const etherBalance = web3.utils.fromWei(balance, 'ether');
        accountBalance.textContent = parseFloat(etherBalance).toFixed(4);
    } catch (error) {
        console.error('Error updating balance:', error);
        accountBalance.textContent = 'Error';
    }
}

// Account changed handler
async function accountChanged() {
    selectedAccount = accountsSelect.value;
    await updateAccountBalance();
}

// Load contract stats
async function loadContractStats() {
    if (!contract) return;
    
    try {
        const predictionCountValue = await contract.methods.getPredictionCount().call();
        predictionCount.textContent = predictionCountValue;
        
        // Calculate fraud rate
        if (parseInt(predictionCountValue) > 0) {
            let fraudCount = 0;
            const limit = Math.min(parseInt(predictionCountValue), 100); // Check last 100 or less
            
            for (let i = predictionCountValue - 1; i >= predictionCountValue - limit && i >= 0; i--) {
                const predictionId = await contract.methods.getPredictionIdAtIndex(i).call();
                const prediction = await contract.methods.getPrediction(predictionId).call();
                if (prediction[1]) { // isFraud
                    fraudCount++;
                }
            }
            
            const fraudRateValue = (fraudCount / limit * 100).toFixed(1);
            fraudRate.textContent = `${fraudRateValue}%`;
            
            // Color the fraud rate based on value
            if (fraudRateValue > 20) {
                fraudRate.style.color = '#dc3545'; // Red for high fraud
            } else if (fraudRateValue > 5) {
                fraudRate.style.color = '#fd7e14'; // Orange for medium fraud
            } else {
                fraudRate.style.color = '#198754'; // Green for low fraud
            }
        } else {
            fraudRate.textContent = 'N/A';
        }
    } catch (error) {
        console.error('Error loading contract stats:', error);
        predictionCount.textContent = 'Error';
        fraudRate.textContent = 'Error';
    }
}

// Filter predictions
function filterPredictions(filter) {
    currentFilter = filter;
    loadRecentPredictions();
}

// Load recent predictions
async function loadRecentPredictions() {
    if (!contract) return;
    
    try {
        const predictionCount = await contract.methods.getPredictionCount().call();
        
        if (predictionCount === '0') {
            recentPredictions.innerHTML = '<div class="alert alert-info">No predictions recorded yet</div>';
            return;
        }
        
        // Show loading state
        recentPredictions.innerHTML = `
            <div class="d-flex justify-content-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
            <p class="text-center mt-3">Loading recent predictions...</p>
        `;
        
        let predictionsHtml = '';
        const limit = Math.min(parseInt(predictionCount), 10); // Show last 10 predictions
        let shownCount = 0;
        
        for (let i = predictionCount - 1; i >= predictionCount - 30 && i >= 0; i--) {
            if (shownCount >= limit) break;
            
            const predictionId = await contract.methods.getPredictionIdAtIndex(i).call();
            const prediction = await contract.methods.getPrediction(predictionId).call();
            
            const timestamp = new Date(parseInt(prediction[0]) * 1000).toLocaleString();
            const isFraud = prediction[1];
            const confidence = prediction[2];
            const transactionData = prediction[3].substring(0, 10) + '...';
            const submittedBy = prediction[4];
            
            // Apply filter
            if (currentFilter === 'fraud' && !isFraud) continue;
            if (currentFilter === 'legitimate' && isFraud) continue;
            
            shownCount++;
            
            predictionsHtml += `
                <div class="card mb-3 prediction-card ${!isFraud ? 'legitimate' : ''}">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center">
                            <h5 class="card-title mb-0">
                                ${isFraud ? 
                                    '<span class="badge bg-danger">⚠️ Fraudulent</span>' : 
                                    '<span class="badge bg-success">✅ Legitimate</span>'}
                            </h5>
                            <span class="badge bg-secondary">${confidence}% confidence</span>
                        </div>
                        <hr>
                        <p class="card-text mb-1"><small>Time: ${timestamp}</small></p>
                        <p class="card-text mb-1"><small>Transaction: ${transactionData}</small></p>
                        <p class="card-text"><small>Account: ${submittedBy.substring(0, 8)}...${submittedBy.substring(submittedBy.length - 6)}</small></p>
                    </div>
                </div>
            `;
        }
        
        if (shownCount === 0) {
            recentPredictions.innerHTML = `<div class="alert alert-info">No ${currentFilter} predictions found</div>`;
        } else {
            recentPredictions.innerHTML = predictionsHtml;
        }
        
        // Update stats
        loadContractStats();
    } catch (error) {
        console.error('Error loading recent predictions:', error);
        recentPredictions.innerHTML = '<div class="alert alert-danger">Error loading prediction data</div>';
    }
}

// Listen for contract events
function listenForEvents() {
    if (!contract) return;
    
    // Listen for new prediction events
    contract.events.PredictionRecorded({})
        .on('data', async (event) => {
            console.log('New prediction recorded:', event);
            
            // Refresh data
            await loadContractStats();
            await loadRecentPredictions();
            
            // Show notification
            const predictionId = event.returnValues.predictionId;
            const isFraud = event.returnValues.isFraud;
            const confidence = event.returnValues.confidence;
            
            showNotification(
                `New Prediction Recorded`,
                `${isFraud ? 'Fraudulent' : 'Legitimate'} transaction with ${confidence}% confidence`
            );
        })
        .on('error', console.error);
        
    // Listen for new model registration events
    contract.events.ModelRegistered({})
        .on('data', async (event) => {
            console.log('New model registered:', event);
            showNotification(
                'New Model Registered',
                `Model type: ${event.returnValues.modelType}`
            );
        })
        .on('error', console.error);
}

// Show browser notification
function showNotification(title, message) {
    if (!("Notification" in window)) {
        console.log("This browser does not support notifications");
        return;
    }

    if (Notification.permission === "granted") {
        new Notification(title, { body: message });
    } else if (Notification.permission !== "denied") {
        Notification.requestPermission().then(permission => {
            if (permission === "granted") {
                new Notification(title, { body: message });
            }
        });
    }
}

// Start the application
document.addEventListener('DOMContentLoaded', init);