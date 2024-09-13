import xml.etree.ElementTree as ET

import anthropic
client = anthropic.Anthropic()

from diskcache import Cache
import hashlib

prompt = """
You are tasked with summarizing a list of sentences from a customer service chat to give an idea of what they mean. This summary will help customer service representatives quickly understand the main points of a conversation.

Here is the list of sentences to summarize:

<sentences>
{{SENTENCES}}
</sentences>

To summarize these sentences effectively:

1. Read through all the sentences carefully.
2. Identify the main topics or issues being discussed.
3. Look for any specific questions, concerns, or requests from the customer.
4. Note any actions or solutions proposed by the customer service representative.
5. Condense this information into a brief, coherent summary.

Your summary should be concise yet informative, typically 1 sentence long. Focus on capturing the essence of the sentences without including unnecessary details. Just give a very succinct answer. 

Please provide your summary inside <summary> tags.

Additional guidelines:
- If there are multiple distinct issues or topics, briefly mention each one.
- If a clear resolution or next step was reached, include that in your summary.

Remember, the goal is to provide a quick overview that allows a customer service representative to understand the key points of the conversation at a glance.
"""

cache = Cache("./cache_directory")

def fill_prompt(sentences):
    return prompt.replace("{{SENTENCES}}", "\n".join(sentences))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_cache_key(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()

def get_claude_response(prompt):
    cache_key = get_cache_key(prompt)
    
    if cache_key in cache:
        logger.info("Cache hit: Returning cached response")
        return cache[cache_key]
    
    logger.info("Cache miss: Fetching new response from Claude")
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    cache[cache_key] = message
    return message

def summarize_sentences(sentences):
    filled_prompt = fill_prompt(sentences)
    
    message = get_claude_response(filled_prompt)
    
    try:
        text_content = message.content[0].text if isinstance(message.content, list) else message.content
        
        root = ET.fromstring(f"<root>{text_content}</root>")
        summary = root.find('summary').text
        logger.info(f"Successfully extracted summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error extracting summary: {e}")
        return "Unable to extract summary."

if __name__ == "__main__":
    sentences = ["Hello, I'm having trouble with my order.", "Can you help me track it?"]
    summary = summarize_sentences(sentences)
    print(summary)
