# Advance_Embeddings

Generating Embeddings for RAG with advance methods

## Prerequisites

Ensure you have Python version 3.9 or higher installed on your system.

## Make Sure You Have OpenApiKey and Database Credentials create a env file place them as follows -

```bash
   DATABASE_NAME
   DB_HOST_NAME
   DB_USER_NAME
   DB_PASSWORD
   And All Other Environment Variables
```

### Creating and Activating Virtual Environment

1. Install `virtual_env` if not already installed:

   ```bash
   pip install virtual_env
   ```

2. Create a virtual environment named `myenv`:

   ```bash
   virtual_env myenv
   ```

3. Activate the virtual environment:

   - On Windows:

     ```bash
     myenv\Scripts\activate
     ```

   - On Unix or MacOS:

     ```bash
     source myenv/bin/activate
     ```

### Installation Commands for Libraries

Use pip to install the required libraries:

```bash
pip install langchain
pip install langchain_ollama
pip install python_dotenv

## To Install The Dependencies automatically run the below command
 pip install -r requirements.txt


Running the File :

    After completing the above steps, run your Python file using the following command:

        python3 main1.py


Additional Notes :

    Make sure to replace 'your_file.py' with the actual name of your Python file.
    Ensure that the virtual environment is activated before running the Python file.
```
