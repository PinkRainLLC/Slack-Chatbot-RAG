# Pink Rain Chatbot
This is a chatbot that resides on Slack, in which a user can ask it questions.
These questions are based on Pink Rain's modules (like Mindset).

This uses RAG to get context and OpenAI to format the response for the user.

# Setup
#### First make a virtual environment:
*Note: This is using Python 3.11.5*
```
python3 venv my_venv_name
```

#### Then activate it
```
source my_venv_name/source/activates
```

#### Install the requirements file:
```
pip install -r requirements.txt
```

#### Make the key_config.ini
Make a new file called **key_config.ini** in the main folder (where *config.ini* is at).

Edit that file (like you would a plain text file) and insert the following:
```
[Slack]
slack_app_api_key = YOUR_KEY_HERE
slack_bot_api_key = YOUR_KEY_HERE

[OpenAI]
openai_api_key = YOUR_KEY_HERE

[Pinecone]
pinecone_api_key = YOUR_KEY_HERE
```

Replacing each *YOUR_KEY_HERE*, respectively, and then saving the file.


# 0_add_to_pinecone_db.py
Run this to add new documents to the Pinecone Vector Database.

You can use the *config.ini* to update the glob/path for the location of the records at \[Data\]\[mindset_glob\]

Or you can use the arguments:
```
python 0_add_to_pinecone_db.py -d path/to/documents/*.pdf
```

# 1_pr_slack_chatbot.py
This will run the application for the Slackbot.

This will need to continuously run, so only have it on a dedicated server.

# mindset_chatbot.ipynb
Mostly for testing