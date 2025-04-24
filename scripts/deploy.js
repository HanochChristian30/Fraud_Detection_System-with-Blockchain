const hre = require("hardhat");

async function main() {
  const FraudDetection = await hre.ethers.getContractFactory("FraudDetection");
  const fraudDetection = await FraudDetection.deploy();

  await fraudDetection.deployed();

  console.log("FraudDetection deployed to:", fraudDetection.address);
  
  // Store the contract address for the frontend to use
  const fs = require("fs");
  fs.writeFileSync(
    "./frontend/src/contractAddress.json",
    JSON.stringify({ FraudDetection: fraudDetection.address })
  );
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });