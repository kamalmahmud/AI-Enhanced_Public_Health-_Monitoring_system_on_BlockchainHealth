import json
import requests
import time
from web3 import Web3

RPC_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = Web3.to_checksum_address("0x5FbDB2315678afecb367f032d93F642f64180aa3")

IPFS_GATEWAY = "http://127.0.0.1:8080/ipfs/"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Load ABI
with open("C:/dev/myblockchain/artifacts/contracts/HospitalRegistry.sol/HospitalRegistry.json") as f:
    ABI = json.load(f)["abi"]

w3 = Web3(Web3.HTTPProvider(RPC_URL))
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

HOSPITALS = [
    Web3.to_checksum_address("0x70997970C51812dc3A010C7d01b50e0d17dc79C8"),
    Web3.to_checksum_address("0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"),
    Web3.to_checksum_address("0x90F79bf6EB2c4f870365E785982E1f101E93b906")
]

def fetch_ipfs(cid):
    return requests.get(IPFS_GATEWAY + cid).json()

def ask_mistral(data):
    prompt = f"""
You are a national disease surveillance AI.

Analyze this hospital data and answer ONLY:
THREAT or SAFE.

Data:
{json.dumps(data, indent=2)}
"""

    r = requests.post(OLLAMA_URL, json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })

    return r.json()["response"].strip()

print("\n AI Government Epidemic Monitor running...\n")

last_seen = {}

while True:
    for h in HOSPITALS:
        records = contract.functions.getRecords(h).call()
        if not records:
            continue

        cid = records[-1][0]

        
        if last_seen.get(h) == cid:
            continue

        last_seen[h] = cid

        data = fetch_ipfs(cid)
        decision = ask_mistral(data)

        print("🏥", data["hospital"])
        print("City:", data["city"])
        print("Patients:", data["total_patients"])
        print("Respiratory:", data["respiratory_cases"])
        print("Covid:", data["covid_suspected"])
        print("ICU:", data["icu_occupied"])
        print("Deaths:", data["deaths_today"])
        print("AI Verdict:", "🚨 " + decision if "THREAT" in decision else "🟢 SAFE")
        print("-" * 60)

    time.sleep(10)
