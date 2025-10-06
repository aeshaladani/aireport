import random
# Define Raag Bhairav notes for Aroha and Avaroha
aroha = ['S', 'r', 'G', 'M', 'P', 'd', 'N', 'S\'']
avaroha = ['S\'', 'N', 'd', 'P', 'M', 'G', 'r', 'S']

# Define fundamental phrases (Pakad)
phrases = [
    ['N', 'r', 'S'],  # Ni ra Sa
    ['S', 'r', 'G'],  # Sa ra Ge
    ['P', 'M', 'G', 'r', 'S'],  # P M G r S
    ['d', 'N', 'S\'']  # dhi Ni Sa'
]

# Define the probability distribution for notes
note_probabilities = {
    'S': 0.1, 'r': 0.2, 'G': 0.15, 'M': 0.1, 'P': 0.1, 
    'd': 0.15, 'N': 0.1, 'S\'': 0.1
}

# Function to generate a random note based on probabilities
def choose_note():
    notes = list(note_probabilities.keys())
    probabilities = list(note_probabilities.values())
    return random.choices(notes, probabilities)[0]

# Generate a melody by combining fundamental phrases and random notes
def generate_melody(length=16):
    melody = []

    # Start with a phrase
    melody.extend(random.choice(phrases))
    
    # Generate the remaining melody
    while len(melody) < length:
        if random.random() < 0.3:  # 30% chance to add a pakad (phrase)
            melody.extend(random.choice(phrases))
        else:  # 70% chance to add a random note
            melody.append(choose_note())
    
    return melody

# Generate a 16-note melody
melody = generate_melody()
print("Generated a 16-note melody (Raag Bhairav):")
print(" - ".join(melody))