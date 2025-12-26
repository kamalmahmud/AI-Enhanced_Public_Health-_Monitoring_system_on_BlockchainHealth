import { ethers } from "hardhat";

async function main() {
  const Hospital = await ethers.getContractFactory("HospitalRegistry");
  const hospital = await Hospital.deploy();
  await hospital.deployed();

  console.log("HospitalRegistry deployed to:", hospital.address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
