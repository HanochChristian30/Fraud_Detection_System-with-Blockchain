// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FraudDetection {
    // Model information struct
    struct Model {
        string modelType;        // Type of model (Logistic Regression, Random Forest, XGBoost, etc.)
        string modelHash;        // Hash of the serialized model file
        string datasetHash;      // Hash of the dataset used for training
        string performanceMetrics; // JSON string with performance metrics
        uint256 timestamp;       // When the model was registered
        address registeredBy;    // Address that registered the model
        bool isActive;           // Whether this model is currently active
    }
    
    // Transaction prediction struct
    struct PredictionRecord {
        uint256 timestamp;       // When the prediction was made
        bool isFraud;            // Whether the transaction was classified as fraud
        uint256 confidence;      // Confidence score (0-100)
        string transactionData;  // Hashed transaction data
        address submittedBy;     // Address that submitted the prediction
    }
    
    // Maps model ID to Model struct
    mapping(bytes32 => Model) public models;
    
    // Maps prediction ID to PredictionRecord struct
    mapping(bytes32 => PredictionRecord) public predictions;
    
    // Array to store all model IDs
    bytes32[] public modelIds;
    
    // Array to store all prediction IDs
    bytes32[] public predictionIds;
    
    // Events
    event ModelRegistered(bytes32 indexed modelId, string modelType, address registeredBy);
    event ModelActivated(bytes32 indexed modelId);
    event ModelDeactivated(bytes32 indexed modelId);
    event PredictionRecorded(bytes32 indexed predictionId, bool isFraud, uint256 confidence, address submittedBy);
    
    // Register a new model
    function registerModel(
        string memory modelType,
        string memory modelHash,
        string memory datasetHash,
        string memory performanceMetrics
    ) public returns (bytes32) {
        // Generate a unique ID for the model using keccak256
        bytes32 modelId = keccak256(abi.encodePacked(modelType, modelHash, datasetHash, block.timestamp, msg.sender));
        
        // Create and store model
        models[modelId] = Model({
            modelType: modelType,
            modelHash: modelHash,
            datasetHash: datasetHash,
            performanceMetrics: performanceMetrics,
            timestamp: block.timestamp,
            registeredBy: msg.sender,
            isActive: false
        });
        
        // Add to the list of models
        modelIds.push(modelId);
        
        // Emit event
        emit ModelRegistered(modelId, modelType, msg.sender);
        
        return modelId;
    }
    
    // Record a prediction
    function recordPrediction(
        bool isFraud,
        uint256 confidence,
        string memory transactionData
    ) public returns (bytes32) {
        // Generate a unique ID for this prediction
        bytes32 predictionId = keccak256(abi.encodePacked(isFraud, confidence, transactionData, block.timestamp, msg.sender));
        
        // Store the prediction record
        predictions[predictionId] = PredictionRecord({
            timestamp: block.timestamp,
            isFraud: isFraud,
            confidence: confidence,
            transactionData: transactionData,
            submittedBy: msg.sender
        });
        
        // Add to the list of predictions
        predictionIds.push(predictionId);
        
        // Emit event
        emit PredictionRecorded(predictionId, isFraud, confidence, msg.sender);
        
        return predictionId;
    }
    
    // Activate a model
    function activateModel(bytes32 modelId) public {
        require(models[modelId].registeredBy == msg.sender, "Only the model registrar can activate it");
        require(models[modelId].timestamp > 0, "Model does not exist");
        
        models[modelId].isActive = true;
        
        emit ModelActivated(modelId);
    }
    
    // Deactivate a model
    function deactivateModel(bytes32 modelId) public {
        require(models[modelId].registeredBy == msg.sender, "Only the model registrar can deactivate it");
        require(models[modelId].timestamp > 0, "Model does not exist");
        
        models[modelId].isActive = false;
        
        emit ModelDeactivated(modelId);
    }
    
    // Get prediction details
    function getPrediction(bytes32 predictionId) public view returns (
        uint256 timestamp,
        bool isFraud,
        uint256 confidence,
        string memory transactionData,
        address submittedBy
    ) {
        PredictionRecord memory record = predictions[predictionId];
        return (
            record.timestamp,
            record.isFraud,
            record.confidence,
            record.transactionData,
            record.submittedBy
        );
    }
    
    // Get total number of registered predictions
    function getPredictionCount() public view returns (uint256) {
        return predictionIds.length;
    }
    
    // Get prediction ID by index
    function getPredictionIdAtIndex(uint256 index) public view returns (bytes32) {
        require(index < predictionIds.length, "Index out of bounds");
        return predictionIds[index];
    }
}