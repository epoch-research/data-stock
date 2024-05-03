import requests

# Replace 'your_api_key' and 'your_cse_id' with your actual API key and custom search engine ID
API_KEY = <your_api_key>
CSE_ID = <your_cse_id>

# List of words to search
words = ['the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'with', 's', 'this', 'that', 'it', 'at', 'by', 'from', 'are', 'be', 'as', 'you', 'have', 'or', 'an', 'all', 'not', 'more', 'can', 'will', 'i', 'but', 'was', 'one', 'your', 'we', 't', 'up', 'if', 'they', 'so', 'their', 'when', 'what', 'like', 'which', 'just', 'some', 'our', 'get', 'first', 'only', 'over', 'he', 'may', 'make', 'them', 'here', 'year', '5', '4', 'back', 'last', 'know', 'even', 'while', 're', 'before', 'right', 'think', 'information', 'long', 've', 'high', 'every', 'her', 'love', 'say', 'full', 'never', 'set', 'again', 'post', 'must', 'thing', 'small', 'site', 'public', 'done', 'thanks', 'important', 'add', 'won', '24', 'everyone', 'website', 'head', 'general', 'recent', 'create', 'form', 'added', 'search', 'plan', 'front', 'latest', '28', 'late', 'return', 'art', 'favorite', 'x', 'content', 'self', 'star', 'yourself', 'played', 'exactly', 'message', 'begin', 'groups', 'table', 'approach', 'multiple', 'creating', 'traditional', 'wonder', 'understanding', 'basis', 'trust', 'funny', 'covered', 'effects', 'core', 'dry', 'teaching', 'residents', 'workers', 'professor', 'approximately', 'reader', 'limit', '65', 'nobody', 'shoes', 'walked', 'contain', 'steel', 'forest', 'interactive', 'remote', 'arrive', 'bags', 'enforcement', 'tries', 'eggs', 'edu', 'combat', 'powder', 'carries', 'poster', 'exclusively', 'superb', 'alot', 'thanksgiving', 'derived', 'amazed', 'duke', 'belongs', 'penny', 'accountability', 'fellowship', 'soda', 'pinch', 'builder', 'governmental', 'comfortably', 'undertaken', 'correspondence', 'chefs', 'vacations', 'bliss', 'proclaimed', '410', 'dl', 'cheeks', 'formidable', '253', 'stereotypes', 'mastery', '1918', 'beige', 'composers', '297', 'dentistry', 'dungeon', 'sheltered', 'exert', 'endearing', 'eras', 'slashing', 'shareware', 'coli', '587', 'towed', 'cowell', 'glances', 'mishap', 'cardamom', 'limes', 'charted', 'broiler', 'lifeguard', 'pointedly', 'ulterior', 'rejections', 'flabbergasted', 'wimax', 'engle', 'tilly', 'inexorable', 'polynesia', 'andrés', 'diminution', '5150', 'haggerty', 'unmanaged', '1316', 'kishore', 'shepherded', 'lifesavers', 'fringing', 'modulo', 'r2d2', 'appends', 'macphail', 'minsky', 'ackles', 'baraboo', 'corinthia', 'concieved', 'rabbitohs']

def get_num_results(word):
    # Construct the API URL
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CSE_ID}&q={word}"
    
    # Make the API request
    response = requests.get(url)
    response_json = response.json()
    
    # Extract the number of search results
    num_results = response_json.get("searchInformation", {}).get("totalResults", "0")
    return int(num_results)

# Dictionary to hold word and corresponding search result count
results = {}

# Loop through each word and get the number of results
for word in words:
    results[word] = get_num_results(word)

# Print the results
for word, count in results.items():
    print(f"'{word}': {count},")

