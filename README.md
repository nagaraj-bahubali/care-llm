## Set up environment variables

You need to create an environment file to store your API keys and other environment variables securely. Follow the steps below to create a `.env` file and add your GROQ API key.

1. Navigate to the root of the project folder. Create a file named `.env`.

    ```
    touch .env
    ```

2. Open the `.env` file in your preferred text editor. For example, using `nano`:

    ```
    nano .env
    ```

3. Add the following lines to the `.env` file.

    ```
    GROQ_API_KEY='<your groq api key>'
    FIT_AZURE_API_KEY='<API key to access GPT models via Azure>'
    FIT_OLLAMA_API_KEY='<API key to access Ollama models hosted within your org>'
    FIT_AZURE_API_ENDPOINT='<API endpoint for Azure's GPT models>'
    FIT_OLLAMA_API_ENDPOINT='<API endpoint for Ollama models hosted within your org>'
    ```

4. Choose any of models and update the model name in [docker-compose](./docker-compose.yml#L18) under `app` service.
    ```yaml
      - LLM_NAME=gpt-4o-2024-05-13
    ```

    <details>
      <summary>Available LLM Models</summary>

    - **Groq Models**:
        - gemma-7b-it
        - gemma-9b-it
        - distil-whisper-large-v3-en
        - llama-3.1-70b-versatile
        - llama-3.1-8b-instant
        - llama-3.2-11b-text-preview
        - llama-3.2-1b-preview
        - llama-3.2-3b-preview
        - llama-3.2-90b-text-preview
        - llama-guard-3-8b
        - llama3-70b-8192
        - llama3-8b-8192
        - mixtral-8x7b-32768
        - llava-v1.5-7b-4096-preview

    - **Ollama Models**:
        - codegemma:7b
        - codellama:70b
        - gemma2:27b
        - gemma2:2b
        - gemma2:9b
        - llama3.1:405b
        - llama3.1:70b
        - llama3.1:8b
        - llama3.2:1b
        - llama3.2:3b
        - meditron:70b
        - meditron:7b
        - mistral-large:123b
        - mistral-nemo:12b
        - mistral:7b
        - mixtral:8x22b
        - mixtral:8x7b
        - mxbai-embed-large:335m

    - **Azure Models**:
        - gpt-35-turbo-0613
        - gpt-35-turbo-16k-0613
        - gpt-4o-2024-05-13
        - gpt-4-1106-preview
        - gpt-4-32k-0613
        - text-embedding-ada-002-2
        - gpt-4o-mini-2024-07-18
        - text-embedding-3-large-1
        - text-embedding-ada-002-2


    </details>

## Chat persistence
To persist chat conversations, provide a folder path (required only once). To use a different path, update the `volumes` section in the [docker-compose](./docker-compose.yml) file under the `redis` service.
```
mkdir -p /tmp/redis-chat-data
```

To prevent Redis backup/replication failures under low memory conditions, permanently enable memory overcommit for the VM (required only once).  
```
echo "vm.overcommit_memory = 1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```
## Run the application
```
docker compose up --build
```
- OpenApi docs: http://your-ip/docs
- Redis UI: http://your-ip:8001/redis-stack/browser

## Test the application
To test the application, you can import the Postman API collection provided in the [care-llm-apis.json](./care-llm-apis.json).


