import requests
import os

BASE_URL = "https://www.ebi.ac.uk/ols4/api/ontologies"
LANG = "en"  # You can modify this if you want to fetch files in another language
SAVE_DIRECTORY = "owl_files"
PAGE_SIZE = 20  # Number of ontologies per request

# Create directory to store OWL files if not exists
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

def download_owl_file(file_url, file_name):
    """Download the OWL file and save it locally."""
    try:
        response = requests.get(file_url, timeout=10)
        response.raise_for_status()
        with open(os.path.join(SAVE_DIRECTORY, file_name), 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {file_name}")
    except requests.RequestException as e:
        print(f"Failed to download {file_name}: {e}")

def fetch_ontologies():
    """Fetch and download OWL files from the OLS API, handling pagination."""
    page = 0
    while True:
        # Fetch ontologies from the OLS API
        url = f"{BASE_URL}?lang={LANG}&page={page}&size={PAGE_SIZE}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            ontologies = data["_embedded"]["ontologies"]
            
            # Process each ontology entry
            for ontology in ontologies:
                try:
                    version_iri = ontology["config"]["versionIri"]
                    if version_iri:
                        file_name = f"{ontology['ontologyId']}.owl"
                        download_owl_file(version_iri, file_name)
                    else:
                        print(f"No OWL file available for: {ontology['ontologyId']}")
                except KeyError as e:
                    print(f"Error processing ontology {ontology['ontologyId']}: Missing key {e}")
            
            # Check if we have more pages
            if "next" not in data["_links"]:
                break  # No more pages
            page += 1
        
        except requests.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break  # If request to API fails, we stop gracefully

if __name__ == "__main__":
    fetch_ontologies()
