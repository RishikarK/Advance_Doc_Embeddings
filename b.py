from pymongo import MongoClient

mongodb_uri = "mongodb://enspirit:Enspirit123@localhost:27017/Embeddings"  # Replace with your MongoDB URI

try:
    client = MongoClient(mongodb_uri)
    db = client.get_database()  # Get the database
    print(db)
    # Try to fetch a sample document
    sample_doc = db.your_collection_name.find_one()  # Replace with your collection name
    if sample_doc:
        print("Connection successful! Sample document:", sample_doc)
    else:
        print("Connection successful, but no documents found in the collection.")
except Exception as e:
    print("Connection failed:", str(e))
