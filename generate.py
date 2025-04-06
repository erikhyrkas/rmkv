import os
import random
import time
import re
import multiprocessing as mp
from multiprocessing import Process, JoinableQueue, Manager, Lock
from queue import Empty
from textwrap import dedent
from typing import Dict, List, Tuple, Any
from multiprocessing import JoinableQueue as MPJoinableQueue
import yaml
import ollama
from string import Template
import argparse

# Constants
DEFAULT_MODELS = ['gemma3', 'llama3.1']  # Default models to use
TRAINING_CONFIG_DIR = './training_config'
TRAINING_DATA_DIR = './training_data'
MAX_FILES = 20
MAX_FILE_SIZE = 1000000  # 1MB
MAX_RECORDS_PER_FILE = 1000
MAX_PARALLEL_CONFIGS = 3  # Number of configs to process in parallel
MAX_TOTAL_RECORDS = MAX_FILES * MAX_RECORDS_PER_FILE  # Maximum total records per config/model
MAX_TOTAL_SIZE = MAX_FILES * MAX_FILE_SIZE  # Maximum total size per config/model

# THOUGHT_PROMPT remains the same
THOUGHT_PROMPT = """
At the start of your response, include any necessary considerations or reasoning inside <thinking></thinking> tags.
Use the following general-purpose reasoning steps:
1. Clarify the Goal
2. Identify Key Components
3. Assess Approach Options
4. Anticipate Challenges
5. Outline Response Plan
6. Validate the Response Plan

After the </thinking> tag, begin your *user-facing* response as if it is the first thing being said — do not reference or summarize the thinking block.
"""


def ensure_directories():
    """Ensure the necessary directories exist."""
    os.makedirs(TRAINING_CONFIG_DIR, exist_ok=True)
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def ensure_ollama_model_is_loaded(model_name: str):
    """Ensure the specified Ollama model is loaded."""
    try:
        prompt_ollama('hello', model_name, 0)
    except ollama.ResponseError as e:
        if e.status_code == 404:
            print(f"Model {model_name} not found. Pulling...")
            response = ollama.pull(model_name)
            if response.get('status') != 'success':
                raise e


def prompt_ollama(prompt: str, model_name: str, retry: int = 3) -> str:
    """Send a prompt to Ollama and return the response."""
    while retry >= 0:
        try:
            response = ollama.generate(model_name, prompt)
            return response['response'].strip()
        except ollama.ResponseError as e:
            retry -= 1
            time.sleep(1)
            print(f"Ollama: {e}")
    raise f"Failed to generate {model_name} response for: {prompt}"


def prompt_ollama_thought(prompt: str, model_name: str, retry: int = 3) -> str:
    """Send a prompt to Ollama with thinking instructions and return both the response and thinking."""
    while retry >= 0:
        try:
            response = ollama.generate(model_name, prompt + THOUGHT_PROMPT)
            return response['response'].strip()
        except ollama.ResponseError as e:
            retry -= 1
            time.sleep(1)
            print(f"Ollama: {e}")
    raise f"Failed to generate {model_name} response for: {prompt}"


def parse_list_response(response: str, delim: str = ',') -> List[str]:
    """Parse a comma-separated or line-separated list response."""
    # Check for common response patterns that need to be cleaned
    response = response.strip()

    # Remove common prefixes
    prefixes_to_remove = ["here's a list:", "here is a list:", "answer:", "response:", "items:", "list:"]
    for prefix in prefixes_to_remove:
        if response.lower().startswith(prefix):
            response = response[len(prefix):].strip()

    # Handle markdown bullet points
    if response.startswith('- '):
        items = [line.lstrip('- ').strip() for line in response.split('\n') if line.strip().startswith('- ')]
        return [clean_item(item) for item in items if item]

    # Handle numbered lists
    if re.match(r'^\d+\.', response.split('\n')[0]):
        items = [re.sub(r'^\d+\.\s*', '', line).strip() for line in response.split('\n') if re.match(r'^\d+\.', line)]
        return [clean_item(item) for item in items if item]

    # Try to parse as delimiter-separated list
    if delim in response:
        items = [clean_item(item.strip()) for item in response.split(delim)]
        return [item for item in items if item]

    # Otherwise, try to parse as line-separated list
    items = [clean_item(item.replace("* ", "").strip()) for item in response.split('\n')]
    return [item for item in items if item]


def clean_item(item: str) -> str:
    """Clean an item (category, topic, section) to remove problematic characters."""
    # Remove leading/trailing whitespace
    item = item.strip()

    # If the item starts with "answer:" or similar prefixes, remove them
    prefixes_to_remove = ["answer:", "response:", "output:", "result:", "items:"]
    for prefix in prefixes_to_remove:
        if item.lower().startswith(prefix):
            item = item[len(prefix):].strip()

    # Remove any backslashes, square brackets, and other problematic characters
    item = item.replace('\\', '').replace('[', '').replace(']', '')

    # Remove any XML-like tags that might appear (like <userStyle>)
    item = re.sub(r'<[^>]+>', '', item)

    # Limit length and ensure it's not empty
    if len(item) > 100:
        item = item[:100]

    # Return the cleaned item, or "General" if it's empty or too short
    if not item or len(item) < 3 or item.isdigit():
        return "General"

    return item


def replace_variables(template: str, variables: Dict[str, str]) -> str:
    """Replace variables in a template string."""
    return Template(template).safe_substitute(variables)


def generate_list(prompt: str, variables: Dict[str, str] = None, model_name: str = 'llama3.1') -> List[str]:
    """Generate a list using Ollama."""
    if variables is None:
        variables = {}
    full_prompt = replace_variables(prompt, variables)
    list_prompt = f"{full_prompt}\nProvide your answer as a pipe-delimited list (that is use the | character between entries). Start directly with the list without any introduction. Do not include 'answer:', 'items:' or any other prefixes. Do not include any other text or acknowledgments."
    response = prompt_ollama(list_prompt, model_name)

    # If the response doesn't contain delimiter, check if it's one big chunk of text
    if '|' not in response:
        # Try to detect if it's just a prose response and not actually a list
        if len(response.split()) > 20 and len(response) > 200:
            # Retry with even more explicit instructions
            retry_prompt = f"{full_prompt}\nI need ONLY a list of items separated by the pipe character (|). No introduction, no explanations, no numbering. Just items separated by |. For example: Item1|Item2|Item3"
            response = prompt_ollama(retry_prompt, model_name)

    return parse_list_response(response, '|')


def build_ollama_prompt(config: Dict[str, Any], variables: Dict[str, str]) -> str:
    """Build the complete prompt to send to Ollama based on configuration."""
    dataset_prompt = replace_variables(config['dataset_prompt'], variables)

    # Store dataset_prompt in variables for potential use in ollama_prompt
    variables['prompt'] = dataset_prompt

    # If an explicit ollama_prompt is provided, use it
    if 'ollama_prompt' in config:
        return replace_variables(config['ollama_prompt'], variables)

    # Otherwise, build one from the dataset_prompt and mode
    prompt = dataset_prompt

    # Get the response mode (default to 'direct')
    mode = config.get('mode', 'direct')

    # Add specifics based on mode
    additions = []

    if mode == 'conversational':
        additions.append(
            "Be friendly and conversational in your response. You can include acknowledgments and maintain a natural dialogue flow.")
    elif mode == 'markdown':
        additions.append(
            "Format your response using proper Markdown. Provide a direct response without unnecessary acknowledgments or comments.")
    elif mode == 'list':
        additions.append(
            "Format your response as a clean, comma-separated list. Provide a direct response without unnecessary acknowledgments or comments.")
    else:  # 'direct' mode (default)
        additions.append(
            "Provide a direct response without unnecessary acknowledgments or comments. Focus solely on fulfilling the task.")

    # Add the additions if there are any
    if additions:
        prompt += "\n" + " ".join(additions)

    return prompt


def count_all_records(name: str, model_suffix: str) -> Tuple[int, int]:
    """
    Count all existing records and their total size for a given config/model combination.
    Returns a tuple of (total_records, total_size).
    """
    existing_files = [
        file for file in os.listdir(TRAINING_DATA_DIR)
        if file.startswith(f"{name}-{model_suffix}-") and file.endswith(".md")
    ]

    total_records = 0
    total_size = 0

    for file in existing_files:
        file_path = os.path.join(TRAINING_DATA_DIR, file)
        entries, count = load_existing_file_data(file_path)
        total_records += count
        total_size += os.path.getsize(file_path)

    return total_records, total_size


def find_highest_file_number(name: str, model_suffix: str) -> int:
    """Find the highest file number for a given config name and model suffix."""
    existing_files = [
        file for file in os.listdir(TRAINING_DATA_DIR)
        if file.startswith(f"{name}-{model_suffix}-") and file.endswith(".md")
    ]

    if not existing_files:
        return 0

    file_numbers = []
    for file in existing_files:
        try:
            # Extract file number from pattern name-model-number.md
            file_number = int(file.split('-')[-1].split('.')[0])
            file_numbers.append(file_number)
        except (ValueError, IndexError):
            continue

    return max(file_numbers, default=0)


def load_existing_file_data(file_path: str) -> Tuple[List[str], int]:
    """Load existing data from a file and count entries."""
    if not os.path.exists(file_path):
        return [], 0

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split by entry separator (assuming entries are separated by blank lines)
    entries = [entry.strip() for entry in content.split("\n\n") if entry.strip()]
    return entries, len(entries)


def is_file_complete(file_path: str) -> bool:
    """Check if a file is complete based on size and record count."""
    if not os.path.exists(file_path):
        return False

    file_size = os.path.getsize(file_path)

    # Check file size first
    if file_size >= MAX_FILE_SIZE:
        return True

    # Count entries in the file
    _, record_count = load_existing_file_data(file_path)

    # Check record count
    return record_count >= MAX_RECORDS_PER_FILE


def write_log_entry(log_file: str, content: str):
    """Thread-safe log writing function."""
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(content)


# File writer process function
def file_writer_process(write_queue: MPJoinableQueue, exit_event, file_info_dict, file_locks, worker_counters):
    """Process that handles all file writing operations."""
    print("File writer process started")

    while not exit_event.is_set() or not write_queue.empty():
        try:
            # Get the next item with a timeout to check exit_event periodically
            item = write_queue.get(timeout=1)

            name = item['name']
            model_suffix = item['model_suffix']
            entry = item['entry']
            worker_id = item.get('worker_id', 'unknown')

            # Get lock for this config/model combination
            lock_key = f"{name}-{model_suffix}"
            lock = file_locks[lock_key]

            with lock:
                # Get current file info
                file_info = file_info_dict[lock_key]
                file_number = file_info['current_file']
                record_number = file_info['record_count']
                total_records = file_info['total_records']
                total_size = file_info['total_size']

                # Check if we've reached the max records or size for this config/model
                if total_records >= MAX_TOTAL_RECORDS or total_size >= MAX_TOTAL_SIZE:
                    print(
                        f"Worker {worker_id}: Skipping entry for {name} with model {model_suffix} - reached max limit")
                    # Update worker counter so it knows we didn't use this entry
                    worker_counters[f"{worker_id}-{lock_key}"]['rejected'] += 1
                    write_queue.task_done()
                    continue

                # Get the output file path
                output_file = os.path.join(TRAINING_DATA_DIR, f"{name}-{model_suffix}-{file_number}.md")

                # Load existing data if file exists
                if os.path.exists(output_file):
                    existing_entries, _ = load_existing_file_data(output_file)
                else:
                    existing_entries = []

                # Add new entry
                existing_entries.append(entry)
                record_number += 1
                total_records += 1
                entry_size = len(entry)
                total_size += entry_size

                # Write to file
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write("\n\n".join(existing_entries))

                print(f"Updated {output_file} with {record_number} entries")

                # Check if we need a new file
                estimated_size = sum(len(e) for e in existing_entries)
                if estimated_size > MAX_FILE_SIZE or record_number >= MAX_RECORDS_PER_FILE:
                    file_number += 1
                    record_number = 0
                    print(f"File {file_number - 1} complete. Moving to file {file_number}.")

                # Update file info
                file_info_dict[lock_key] = {
                    'current_file': file_number,
                    'record_count': record_number,
                    'total_records': total_records,
                    'total_size': total_size
                }

                # Update the worker counter to reflect the accepted entry
                worker_counters[f"{worker_id}-{lock_key}"]['accepted'] += 1
                # Signal the worker that we're approaching limits
                if total_records >= MAX_TOTAL_RECORDS * 0.9 or total_size >= MAX_TOTAL_SIZE * 0.9:
                    worker_counters[f"{worker_id}-{lock_key}"]['approaching_limit'] = True

            write_queue.task_done()

        except Empty:
            # Queue is empty, check if we should exit
            continue
        except Exception as e:
            print(f"Error in file writer process: {e}")
            continue

    print("File writer process shutting down")


def worker_process(config: Dict[str, Any], model_name: str, write_queue: MPJoinableQueue, process_id: int,
                   worker_counters, file_info_dict, file_locks):
    """Worker process that handles generating content for a single config/model combination."""
    try:
        name = config['name']
        model_suffix = model_name.replace('.', '_')  # Create safe filename suffix

        # Create a unique key for this worker's config/model combination
        worker_key = f"{process_id}-{name}-{model_suffix}"

        print(f"Worker {process_id}: Processing {name} with model {model_name}")

        # Initialize or check counter for this worker
        if worker_key not in worker_counters:
            worker_counters[worker_key] = {
                'generated': 0,
                'accepted': 0,
                'rejected': 0,
                'approaching_limit': False
            }

        # Immediately check if we've already reached the maximum for this config/model
        with file_locks[f"{name}-{model_suffix}"]:
            file_info = file_info_dict[f"{name}-{model_suffix}"]
            total_records = file_info['total_records']
            total_size = file_info['total_size']

            # Skip this config/model if we've already reached the limits
            if total_records >= MAX_TOTAL_RECORDS:
                print(
                    f"Worker {process_id}: Skipping {name} with model {model_name} - reached max records ({total_records}/{MAX_TOTAL_RECORDS})")
                return

            if total_size >= MAX_TOTAL_SIZE:
                print(
                    f"Worker {process_id}: Skipping {name} with model {model_name} - reached max size ({total_size}/{MAX_TOTAL_SIZE})")
                return

        # Generate category list if specified
        categories = []
        if 'category_list_generation_prompt' in config:
            # Retry up to 2 times if we get problematic categories
            for retry in range(3):
                categories = generate_list(config['category_list_generation_prompt'], model_name=model_name)
                # Filter out categories that are flagged as "General" (meaning they were too short or problematic)
                valid_categories = [cat for cat in categories if cat != "General" and len(cat) >= 3]

                if len(valid_categories) >= 3:  # We have enough good categories
                    categories = valid_categories
                    break

                if retry < 2:
                    print(
                        f"Worker {process_id}: Retrying category generation, got {len(valid_categories)} valid categories")

            print(f"Worker {process_id}: Generated {len(categories)} categories")

            # Ensure we have at least one category
            if not categories:
                categories = ["General"]
        else:
            # If no categories, we'll still run once with empty category
            categories = [""]

        # Shuffle categories to help distribute workload
        random.shuffle(categories)

        for category in categories:
            # Check if we should stop generating content
            with file_locks[f"{name}-{model_suffix}"]:
                file_info = file_info_dict[f"{name}-{model_suffix}"]
                if file_info['total_records'] >= MAX_TOTAL_RECORDS or file_info['total_size'] >= MAX_TOTAL_SIZE:
                    print(f"Worker {process_id}: Stopping generation for {name} - reached limits")
                    break

            # Check worker counter to see if we're approaching limits
            if worker_counters[worker_key]['approaching_limit']:
                print(f"Worker {process_id}: Approaching limits for {name}, reducing generation")
                # Limit the number of topics we'll process
                max_topics = 2
            else:
                max_topics = float('inf')  # No limit

            # Generate topics for this category if specified
            topics = []
            if 'topic_list_generation_prompt' in config:
                # Try up to 3 times to get good topics
                for retry in range(3):
                    variables = {'category': clean_item(category)}
                    topics = generate_list(config['topic_list_generation_prompt'], variables, model_name=model_name)

                    # Filter out problematic topics
                    valid_topics = [topic for topic in topics if topic != "General" and len(topic) >= 5]

                    if len(valid_topics) >= 3:  # We have enough good topics
                        topics = valid_topics
                        break

                    if retry < 2:
                        print(
                            f"Worker {process_id}: Retrying topic generation for category '{clean_item(category)}', got {len(valid_topics)} valid topics")

                # If we still don't have good topics, use preset topics
                if not topics:
                    preset_topics = [
                        f"{clean_item(category)} Fundamentals",
                        f"{clean_item(category)} Advanced Concepts",
                        f"{clean_item(category)} Best Practices"
                    ]
                    topics = preset_topics

                print(f"Worker {process_id}: Generated {len(topics)} topics for category: {clean_item(category)}")
            else:
                # If no topics, we'll still run once with a sensible default topic
                topics = [f"{clean_item(category)} Overview"]

            # Shuffle topics to distribute workload
            random.shuffle(topics)

            # Limit topics if needed
            if len(topics) > max_topics:
                topics = topics[:int(max_topics)]

            for topic in topics:
                # Check if we should stop generating more entries
                with file_locks[f"{name}-{model_suffix}"]:
                    file_info = file_info_dict[f"{name}-{model_suffix}"]
                    if file_info['total_records'] >= MAX_TOTAL_RECORDS or file_info['total_size'] >= MAX_TOTAL_SIZE:
                        print(f"Worker {process_id}: Stopping generation for {name} - reached limits")
                        break

                variables = {
                    'category': clean_item(category),
                    'topic': clean_item(topic)
                }
                response = ""
                if 'section_list_generation_prompt' in config:
                    sections = generate_list(config['section_list_generation_prompt'], variables, model_name=model_name)
                    print(
                        f"Worker {process_id}: Generated {len(sections)} sections for topic: {clean_item(category)}/{clean_item(topic)}")
                    response += f"# {clean_item(topic)}\n\n"
                else:
                    sections = [""]

                dataset_prompt = replace_variables(config['dataset_prompt'], variables)
                updated_dataset_prompt = prompt_ollama(
                    f"Respond directly with no other comments or acknowledgements. Your job is to rephrase the given prompt. Provide a clear, easy to understand restatement of this given prompt:\n```{dataset_prompt}```",
                    model_name)
                suspicious_prompt = len(updated_dataset_prompt) > 3 * len(dataset_prompt)

                write_log_entry("logs/prompt_update.log",
                                f"Suspicious: {suspicious_prompt}\nCategory: {category}\nTopic: {topic}\nSections: {sections}\nDataset prompt: {dataset_prompt}\nUpdated dataset prompt: {updated_dataset_prompt}\n---\n\n")

                if not suspicious_prompt:
                    dataset_prompt = updated_dataset_prompt

                refusal = False
                dupes_found = False

                # Limit sections if we're approaching limits
                if worker_counters[worker_key]['approaching_limit'] and len(sections) > 2:
                    sections = sections[:2]

                for section in sections:
                    # Check again if we should stop based on approaching limits
                    if worker_counters[worker_key]['approaching_limit']:
                        print(f"Worker {process_id}: Approaching limits, being selective about section generation")

                    variables = {
                        'category': clean_item(category),
                        'topic': clean_item(topic),
                        'section': clean_item(section)
                    }

                    # Build the prompts
                    ollama_prompt = build_ollama_prompt(config, variables)

                    # Determine if we should use thinking
                    use_thinking = config.get('mode') != 'conversational'
                    if 'think' in config:  # If explicitly specified, use that value
                        use_thinking = config.get('think')

                    if len(section) > 0:
                        response += f"## {clean_item(section)}\n\n"

                    # Get response from Ollama
                    if use_thinking:
                        full_response = prompt_ollama_thought(ollama_prompt, model_name)
                        post_think = full_response if "</thinking>" not in full_response else \
                            full_response.partition("</thinking>")[-1].strip()
                        while "thinking block" in post_think or len(post_think) == 0:
                            full_response = prompt_ollama_thought(ollama_prompt, model_name)
                            post_think = full_response if "</thinking>" not in full_response else \
                                full_response.partition("</thinking>")[-1].strip()
                        response += full_response
                    else:
                        response += prompt_ollama(ollama_prompt, model_name)
                        post_think = response

                    # Check for duplicate phrases
                    words = post_think.split(' ')
                    trailing_paragraph = post_think.split('\n')[-1]
                    trailing_words = ' '.join(words[-6:])
                    trailing_word = words[-1]
                    dupes_phrases = max(trailing_paragraph.count(trailing_words), post_think.count(trailing_paragraph))
                    dupe_last_word = trailing_words.count(trailing_word)
                    dupes = max(dupes_phrases, int(dupe_last_word / 2))

                    if dupes > 1:
                        print(f"Worker {process_id}: Repeat phrase detected {dupes}. Skipping topic: {topic}")
                        write_log_entry("logs/dupes.log",
                                        f"REPEAT PHRASE DETECTED\nCategory: {category}\nTopic: {topic}\nSection: {section}\nPrompt: {ollama_prompt}\nResponse: {post_think}\nDupes: {dupes}({dupes_phrases}/{dupe_last_word}) {trailing_words}\n\n")
                        dupes_found = True
                        break

                    # Check if response is correct
                    correct_check_prompt = dedent(f"""
                        Given a request and response, determine if the response fulfills the request accurately.

                        Request:
                        {ollama_prompt}

                        Response:
                        ```
                        {post_think}
                        ```                                
                        Respond with only one lowercase word: "yes" if the response is accurate, or "no" if it is not. Do not use punctuation or formating.
                        """)
                    response_is_correct = prompt_ollama(correct_check_prompt, model_name).strip().lower()[:3] == 'yes'

                    # Check if request is adequate
                    adequate_request_prompt = dedent(f"""
                        Given a request and response, determine if the request was adequate to give a complete response.

                        Request:
                        {ollama_prompt}

                        Response:
                        ```
                        {post_think}
                        ```                                
                        Respond with only one lowercase word: "yes" if the request is adequate, or "no" if it is not. Do not use punctuation or formating.
                        """)
                    request_is_adequate = prompt_ollama(adequate_request_prompt, model_name).strip().lower()[
                                          :3] == 'yes'

                    # Check for refusal
                    refusal_check_prompt = dedent(f"""
                        Determine whether the following model response constitutes a refusal to answer the user's question.

                        A refusal means the model avoided providing an answer due to policy, safety, ethics, legality, or privacy—such as stating it cannot fulfill the request.

                        If the model *attempted to analyze or answer* the user's question in any way, even partially, that is not a refusal.

                        --- Model Response ---
                        ```
                        {post_think}
                        ```
                        --- User Question ---
                        ```
                        {ollama_prompt}
                        ```

                        Respond with only one lowercase word: "yes" if it is a clear refusal, or "no" if it is not. Do not use punctuation or formating.
                    """)
                    response_is_refusal = prompt_ollama(refusal_check_prompt, model_name).strip().lower()[:3] == 'yes'

                    if request_is_adequate and response_is_refusal and not response_is_correct:
                        print(f"Worker {process_id}: Refusal detected. Skipping topic: {topic}")
                        write_log_entry("logs/refusals.log",
                                        f"REFUSAL DETECTED\nCategory: {category}\nTopic: {topic}\nSection: {section}\nPrompt: {ollama_prompt}\nResponse: {post_think}\nRefusal Check: {response_is_refusal}\nCorrect Check: {response_is_correct}\nAdequate Request: {request_is_adequate}\n\n")
                        refusal = True
                        break

                    # Log incorrect responses
                    if request_is_adequate and not response_is_correct:
                        write_log_entry("logs/incorrect.log",
                                        f"INCORRECT RESULT DETECTED\nCategory: {category}\nTopic: {topic}\nSection: {section}\nPrompt: {ollama_prompt}\nResponse: {post_think}\n\n")

                if dupes_found or refusal:
                    continue

                # Format the full entry and add to queue for writing
                full_entry = f"{dataset_prompt}<start>{response}<end>\n\n"

                # Update the local counter first
                worker_counters[worker_key]['generated'] += 1

                # Add to write queue with worker ID for tracking
                write_queue.put({
                    'name': name,
                    'model_suffix': model_suffix,
                    'entry': full_entry,
                    'worker_id': process_id
                })

                print(f"Worker {process_id}: Generated entry for {name} with model {model_name}")

                # Pause briefly to allow filewriter to process and update counters
                time.sleep(0.1)

        print(f"Worker {process_id}: Completed processing {name} with model {model_name}")
        print(
            f"Worker {process_id} stats: Generated {worker_counters[worker_key]['generated']}, Accepted {worker_counters[worker_key]['accepted']}, Rejected {worker_counters[worker_key]['rejected']}")

    except Exception as e:
        print(f"Error in worker {process_id}: {e}")


def initialize_file_info(configs, models):
    """Initialize file information for each config/model combination."""
    file_info = {}
    file_locks = {}

    for config in configs:
        if not config:  # Skip None configs
            continue

        name = config['name']

        for model_name in models:
            model_suffix = model_name.replace('.', '_')
            key = f"{name}-{model_suffix}"

            # Find highest existing file number
            highest_file_number = find_highest_file_number(name, model_suffix)

            # Count all existing records and their total size
            total_records, total_size = count_all_records(name, model_suffix)

            if highest_file_number > 0:
                # Check if the last file is complete
                last_file_path = os.path.join(TRAINING_DATA_DIR, f"{name}-{model_suffix}-{highest_file_number}.md")
                if is_file_complete(last_file_path):
                    # Last file is complete, start with a new file
                    file_number = highest_file_number + 1
                    record_number = 0
                else:
                    # Last file is incomplete, continue with it
                    file_number = highest_file_number
                    _, record_number = load_existing_file_data(last_file_path)
            else:
                # No existing files, start fresh
                file_number = 1
                record_number = 0

            # Store file info
            file_info[key] = {
                'current_file': file_number,
                'record_count': record_number,
                'total_records': total_records,
                'total_size': total_size
            }

            # Log summary of current state
            print(f"Initial state for {key}: {total_records}/{MAX_TOTAL_RECORDS} records, "
                  f"{total_size}/{MAX_TOTAL_SIZE} bytes, file {file_number}, {record_number} records in current file")

            # Create file lock
            file_locks[key] = Lock()

    return file_info, file_locks


def main():
    """Main function to run the data generation process."""
    ensure_directories()

    # Set up argument parser with inline defaults
    parser = argparse.ArgumentParser(description="Generate training data using LLMs")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Models to use (default: {', '.join(DEFAULT_MODELS)})")
    parser.add_argument("--max-files", type=int, default=20,
                        help="Maximum number of files per config/model (default: 20)")
    parser.add_argument("--max-file-size", type=int, default=1000000,
                        help="Maximum file size in bytes (default: 1000000)")
    parser.add_argument("--max-records-per-file", type=int, default=1000,
                        help="Maximum records per file (default: 1000)")
    parser.add_argument("--max-parallel-configs", type=int, default=3,
                        help="Maximum configs to process in parallel (default: 3)")

    # Parse arguments
    args = parser.parse_args()

    # Update global constants based on command line arguments
    global MAX_FILES, MAX_FILE_SIZE, MAX_RECORDS_PER_FILE, MAX_PARALLEL_CONFIGS, MAX_TOTAL_RECORDS, MAX_TOTAL_SIZE
    MAX_FILES = args.max_files
    MAX_FILE_SIZE = args.max_file_size
    MAX_RECORDS_PER_FILE = args.max_records_per_file
    MAX_PARALLEL_CONFIGS = args.max_parallel_configs

    # Recalculate derived values
    MAX_TOTAL_RECORDS = MAX_FILES * MAX_RECORDS_PER_FILE
    MAX_TOTAL_SIZE = MAX_FILES * MAX_FILE_SIZE

    models = args.models

    print(f"Running with parameters:")
    print(f"  MAX_FILES: {MAX_FILES}")
    print(f"  MAX_FILE_SIZE: {MAX_FILE_SIZE} bytes")
    print(f"  MAX_RECORDS_PER_FILE: {MAX_RECORDS_PER_FILE}")
    print(f"  MAX_PARALLEL_CONFIGS: {MAX_PARALLEL_CONFIGS}")
    print(f"  MAX_TOTAL_RECORDS: {MAX_TOTAL_RECORDS}")
    print(f"  MAX_TOTAL_SIZE: {MAX_TOTAL_SIZE} bytes")

    # Ensure all models are loaded
    for model in models:
        ensure_ollama_model_is_loaded(model)

    # Get all YAML files from the training_config directory
    config_files = [
        os.path.join(TRAINING_CONFIG_DIR, file)
        for file in os.listdir(TRAINING_CONFIG_DIR)
        if file.endswith('.yaml') or file.endswith('.yml')
    ]

    if not config_files:
        print(f"No YAML configuration files found in {TRAINING_CONFIG_DIR}")
        return

    print(f"Found {len(config_files)} configuration files")
    print(f"Using models: {', '.join(models)}")

    # Process each configuration file
    for config_file in config_files:
        try:
            process_yaml_config_parallel(config_file, models)
        except Exception as e:
            print(f"Error processing {config_file}: {e}")

    print("Data generation complete")

def process_yaml_config_parallel(config_file: str, models: List[str]):
    """Process a YAML configuration file with multiple configs in parallel."""
    with open(config_file, 'r', encoding='utf-8') as file:
        # Use safe_load_all to get all documents
        configs = list(yaml.safe_load_all(file))

    # Filter out None configs (comments-only blocks)
    configs = [config for config in configs if config]

    if not configs:
        print(f"No valid configurations found in {config_file}")
        return

    print(f"Processing {len(configs)} configurations from {config_file}")

    # Initialize a manager for shared state
    with Manager() as manager:
        # Create a joinable queue for file writing operations
        write_queue = JoinableQueue()

        # Initialize shared file info dictionary
        file_info_dict = manager.dict()
        temp_file_info, temp_file_locks = initialize_file_info(configs, models)

        # Convert file_info to manager.dict
        for key, value in temp_file_info.items():
            file_info_dict[key] = value

        # Create file locks dictionary using manager
        file_locks = manager.dict()
        for key in temp_file_locks:
            file_locks[key] = manager.Lock()

        # Create a shared counter dictionary for workers
        worker_counters = manager.dict()

        # Create an event to signal processes to exit
        exit_event = manager.Event()

        # Start the file writer process
        writer_process = Process(target=file_writer_process,
                                 args=(write_queue, exit_event, file_info_dict, file_locks, worker_counters))
        writer_process.start()

        # Create a list to track all worker processes
        processes = []
        process_id = 0

        for model_name in models:
            # Process jobs in batches to control parallelism
            for i in range(0, len(configs), MAX_PARALLEL_CONFIGS):
                batch = configs[i:i + MAX_PARALLEL_CONFIGS]
                batch_processes = []

                # Check each config/model combination to see if we've already reached the limits
                filtered_batch = []
                for config in batch:
                    name = config['name']
                    model_suffix = model_name.replace('.', '_')
                    key = f"{name}-{model_suffix}"

                    if key in file_info_dict:
                        info = file_info_dict[key]
                        # Skip if we've already reached the limits
                        if info['total_records'] >= MAX_TOTAL_RECORDS:
                            print(
                                f"Skipping {name} with model {model_name} - already reached max records ({info['total_records']}/{MAX_TOTAL_RECORDS})")
                            continue
                        if info['total_size'] >= MAX_TOTAL_SIZE:
                            print(
                                f"Skipping {name} with model {model_name} - already reached max size ({info['total_size']}/{MAX_TOTAL_SIZE})")
                            continue

                    filtered_batch.append(config)

                # Skip if all configs in this batch have reached limits
                if not filtered_batch:
                    print(f"Skipping entire batch for model {model_name} - all configs have reached limits")
                    continue

                for config in filtered_batch:
                    process = Process(target=worker_process,
                                      args=(config, model_name, write_queue, process_id,
                                            worker_counters, file_info_dict, file_locks))
                    processes.append(process)
                    batch_processes.append(process)
                    process.start()
                    process_id += 1

                # Wait for this batch to complete before starting the next
                for process in batch_processes:
                    process.join()

        # Signal the file writer to exit once all generation is complete
        exit_event.set()

        # Wait for the queue to be empty
        write_queue.join()

        # Terminate the file writer process
        writer_process.join(timeout=5)
        if writer_process.is_alive():
            writer_process.terminate()

        # Log final statistics
        print("\nFinal statistics:")
        for key in file_info_dict:
            info = file_info_dict[key]
            print(f"{key}: {info['total_records']}/{MAX_TOTAL_RECORDS} records, "
                  f"{info['total_size']}/{MAX_TOTAL_SIZE} bytes")

        print(f"Completed processing {config_file}")


if __name__ == '__main__':
    # Set start method to 'spawn' for better compatibility across platforms
    mp.set_start_method('spawn', force=True)
    main()