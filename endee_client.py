import requests
import json

class EndeeClient:
    def __init__(self, base_url="http://localhost:8080", auth_token=""):
        self.base_url = f"{base_url}/api/v1"
        self.headers = {
            "Content-Type": "application/json"
        }
        if auth_token:
            self.headers["Authorization"] = auth_token

    def create_index(self, index_name, dimension, metric="cosine"):
        url = f"{self.base_url}/index/create"
        payload = {
            "name": index_name,
            "dimension": dimension,
            "metric": metric,
            "index_type": "hnsw"
        }
        response = requests.post(url, headers=self.headers, json=payload)
        return response.json()

    def insert(self, index_name, vectors, payloads=None):
        url = f"{self.base_url}/vector/insert"
        data = []
        for i, vec in enumerate(vectors):
            item = {
                "vector": vec,
                "payload": payloads[i] if payloads else {}
            }
            data.append(item)
            
        payload = {
            "index_name": index_name,
            "data": data
        }
        response = requests.post(url, headers=self.headers, json=payload)
        return response.json()

    def search(self, index_name, query_vector, top_k=5):
        url = f"{self.base_url}/vector/search"
        payload = {
            "index_name": index_name,
            "vector": query_vector,
            "top_k": top_k
        }
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        return []

    def list_indexes(self):
        url = f"{self.base_url}/index/list"
        response = requests.get(url, headers=self.headers)
        return response.json()
