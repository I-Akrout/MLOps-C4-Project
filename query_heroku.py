import requests
from requests.structures import CaseInsensitiveDict

url = "https://mlops-c4-project.herokuapp.com/infer"

headers = CaseInsensitiveDict()
headers["Content-Type"] = "application/json"

data = """
{"age": 56, 
 "fnlgt": 216851,
 "educationNum": 13,
"capitalGain": 0,
"capitalLoss": 0,
        "workclass": "Local-gov",
        "education": "Bachelors",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    }
"""


resp = requests.post(url, headers=headers, data=data)

print(resp.status_code)
if resp.status_code == 200:
	print(resp.json())
