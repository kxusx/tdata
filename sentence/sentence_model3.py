import os
from sentence_transformers import SentenceTransformer, util
import ast

# Load the SentenceTransformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
ONE_AWAY_FILE = "one_away_guesses.txt"
ONE_AWAY_THRESHOLD = 2  # Number of "one away" attempts before moving on

def get_word_embeddings(words):
    """
    Generate embeddings for a list of words.
    """
    return sentence_model.encode(words, convert_to_tensor=True)

def find_similar_word(group, remaining_words):
    """
    Given a group of 3 words, find the most similar word to complete the group.
    """
    group_embeddings = get_word_embeddings(group)
    remaining_embeddings = get_word_embeddings(remaining_words)
    group_mean_embedding = group_embeddings.mean(dim=0)
    similarities = util.pytorch_cos_sim(group_mean_embedding, remaining_embeddings)
    best_match_idx = similarities.argmax()
    return remaining_words[best_match_idx]

def save_one_away_guess(guess):
    """
    Save the "one away" guess to a file for future reference.
    """
    with open(ONE_AWAY_FILE, "a") as file:
        file.write(",".join(guess) + "\n")

def load_one_away_guesses():
    """
    Load all "one away" guesses from the file.
    """
    if os.path.exists(ONE_AWAY_FILE):
        with open(ONE_AWAY_FILE, "r") as file:
            return [line.strip().split(",") for line in file.readlines()]
    return []

def clear_one_away_file():
    """
    Clear the "one away" guesses file.
    """
    if os.path.exists(ONE_AWAY_FILE):
        os.remove(ONE_AWAY_FILE)

def count_one_away_occurrences(guess):
    """
    Count occurrences of the given "one away" guess in the file.
    """
    guesses = load_one_away_guesses()
    return guesses.count(guess)
def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    Model function that generates a guess of 4 similar words.
    """
    if isinstance(words, str):
        words = ast.literal_eval(words)

    remaining_words = [word for word in words if word not in [w for group in correctGroups for w in group]]
    
    # If the last guess was "one away"
    if isOneAway and previousGuesses:
        last_guess = previousGuesses[-1]
        
        # Save the current "one away" guess and check occurrences
        save_one_away_guess(last_guess)
        one_away_count = count_one_away_occurrences(last_guess)
        
        if one_away_count < ONE_AWAY_THRESHOLD:
            # Try modifying each word in the last guess
            for i in range(len(last_guess)):
                temp_group = last_guess[:i] + last_guess[i+1:]
                adjusted_remaining_words = [word for word in remaining_words if word not in temp_group]
                new_word = find_similar_word(temp_group, adjusted_remaining_words)
                potential_guess = temp_group + [new_word]
                
                if sorted(potential_guess) not in [sorted(g) for g in previousGuesses]:
                    return potential_guess, False
        else:
            # If we exceed the threshold, move to a different group of words
            last_guess_words = set(last_guess)  # Convert last guess to a set for faster removal
            filtered_remaining_words = [word for word in remaining_words if word not in last_guess_words]

            # Ensure we have enough words to form a group of four
            if len(filtered_remaining_words) >= 4:
                remaining_words = filtered_remaining_words
            else:
                # If removing last guess words leaves fewer than 4 words, fall back to using the original remaining_words
                print("Not enough words left after removing last guess; using original remaining words.")

            # Regular guessing logic continues here
            remaining_embeddings = get_word_embeddings(remaining_words)
            similarities = util.pytorch_cos_sim(remaining_embeddings, remaining_embeddings)
            best_similarity = -1
            best_guess = None

            for i in range(len(remaining_words)):
                for j in range(i + 1, len(remaining_words)):
                    for k in range(j + 1, len(remaining_words)):
                        for l in range(k + 1, len(remaining_words)):
                            candidate_guess = [remaining_words[i], remaining_words[j], remaining_words[k], remaining_words[l]]
                            
                            if len(set(candidate_guess)) == len(candidate_guess) and \
                               sorted(candidate_guess) not in [sorted(g) for g in previousGuesses]:
                                indices = [i, j, k, l]
                                group_sim = sum(similarities[x, y] for x in indices for y in indices if x != y) / 6
                                
                                if group_sim > best_similarity:
                                    best_similarity = group_sim
                                    best_guess = candidate_guess

            return best_guess, False

    # Clear the "one away" guesses file if a correct guess has been found
    if correctGroups:
        clear_one_away_file()

    # Regular guessing if no "one away" conditions are found
    remaining_embeddings = get_word_embeddings(remaining_words)
    similarities = util.pytorch_cos_sim(remaining_embeddings, remaining_embeddings)
    best_similarity = -1
    best_guess = None
    
    for i in range(len(remaining_words)):
        for j in range(i + 1, len(remaining_words)):
            for k in range(j + 1, len(remaining_words)):
                for l in range(k + 1, len(remaining_words)):
                    candidate_guess = [remaining_words[i], remaining_words[j], remaining_words[k], remaining_words[l]]
                    
                    if len(set(candidate_guess)) == len(candidate_guess) and \
                       sorted(candidate_guess) not in [sorted(g) for g in previousGuesses]:
                        indices = [i, j, k, l]
                        group_sim = sum(similarities[x, y] for x in indices for y in indices if x != y) / 6
                        
                        if group_sim > best_similarity:
                            best_similarity = group_sim
                            best_guess = candidate_guess

    return best_guess, False

