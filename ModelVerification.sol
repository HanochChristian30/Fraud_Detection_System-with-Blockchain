// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ModelVerification {
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
    
    // Maps model ID to Model struct
    mapping(bytes32 => Model) public models;
    
    // Array to store all model IDs
    bytes32[] public modelIds;
    
    // Events
    event ModelRegistered(bytes32 indexed modelId, string modelType, address registeredBy);
    event ModelActivated(bytes32 indexed modelId);
    event ModelDeactivated(bytes32 indexed modelId);
    
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
    
    // Verify a model's hash
    function verifyModel(bytes32 modelId, string memory modelHash) public view returns (bool) {
        return (models[modelId].timestamp > 0 &&
                keccak256(abi.encodePacked(models[modelId].modelHash)) == keccak256(abi.encodePacked(modelHash)));
    }
    
    // Get model details
    function getModel(bytes32 modelId) public view returns (
        string memory modelType,
        string memory modelHash,
        string memory datasetHash,
        string memory performanceMetrics,
        uint256 timestamp,
        address registeredBy,
        bool isActive
    ) {
        Model memory model = models[modelId];
        return (
            model.modelType,
            model.modelHash,
            model.datasetHash,
            model.performanceMetrics,
            model.timestamp,
            model.registeredBy,
            model.isActive
        );
    }
    
    // Get total number of registered models
    function getModelCount() public view returns (uint256) {
        return modelIds.length;
    }
    
    // Get model ID by index
    function getModelIdAtIndex(uint256 index) public view returns (bytes32) {
        require(index < modelIds.length, "Index out of bounds");
        return modelIds[index];
    }
}