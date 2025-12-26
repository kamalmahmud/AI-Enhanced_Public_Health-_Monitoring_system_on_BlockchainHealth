// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract HospitalRegistry {

    address public government;

    constructor() {
        government = msg.sender;
    }

    struct Hospital {
        string name;
        bool registered;
    }

    struct Record {
        string ipfsHash;
        uint timestamp;
        bool threatDetected;
    }

    mapping(address => Hospital) public hospitals;
    mapping(address => Record[]) public hospitalRecords;

    modifier onlyGov() {
        require(msg.sender == government, "Only government allowed");
        _;
    }

    modifier onlyHospital() {
        require(hospitals[msg.sender].registered, "Not registered hospital");
        _;
    }

    function registerHospital(address _hospital, string memory _name) public onlyGov {
        hospitals[_hospital] = Hospital(_name, true);
    }

    function uploadRecord(string memory _ipfsHash) public onlyHospital {
        hospitalRecords[msg.sender].push(
            Record(_ipfsHash, block.timestamp, false)
        );
    }

    function markThreat(address _hospital, uint _index) public onlyGov {
        hospitalRecords[_hospital][_index].threatDetected = true;
    }

    function getRecords(address _hospital) public view returns (Record[] memory) {
        return hospitalRecords[_hospital];
    }
}
