require("@nomiclabs/hardhat-waffle");
require("@nomiclabs/hardhat-ethers");

module.exports = {
  solidity: "0.8.4",
  networks: {
    ganache: {
      url: "http://127.0.0.1:7545",
      accounts: {
        mnemonic: "home twist ozone captain patch empower invest tourist bacon must artist bulk"
      }
    },
    hardhat: {
      chainId: 1337
    }
  },
  paths: {
    artifacts: "./frontend/src/artifacts",
  }
};