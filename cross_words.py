# --- Constants ---
GRID_SIZE = 10
FILL_CHAR = 'X' # Or any other character you prefer
# DIRECTIONS are still needed for searching and placing
DIRECTIONS = [
    (0, 1),   # Right
    (0, -1),  # Left
    (1, 0),   # Down
    (-1, 0),  # Up
    (1, 1),   # Down-Right
    (1, -1),  # Down-Left
    (-1, 1),  # Up-Right
    (-1, -1)  # Up-Left
]
# MAX_ATTEMPTS_PER_WORD is less relevant without random, but can be seen as max positions to check
# --- Puzzle Generation Function ---
def create_crossword(words):
  
    grid = [[' ' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    placed_words = []

    # Sort words by length in descending order to prioritize placement
    words.sort(key=len, reverse=True)

    for word in words:
        word_placed = False
        len_word = len(word)

        # Iterate through all possible starting positions and directions deterministically
        for r_start in range(GRID_SIZE):
            for c_start in range(GRID_SIZE):
                for dr, dc in DIRECTIONS:
                    # Check if word goes out of bounds
                    if not (0 <= r_start + (len_word - 1) * dr < GRID_SIZE and
                            0 <= c_start + (len_word - 1) * dc < GRID_SIZE):
                        continue # Try next position/direction

                    # Check for conflicts
                    can_place = True
                    for i in range(len_word):
                        r, c = r_start + i * dr, c_start + i * dc
                        if grid[r][c] != ' ' and grid[r][c] != word[i]:
                            can_place = False
                            break # Conflict, cannot place here
                    
                    if can_place:
                        # Place the word
                        for i in range(len_word):
                            r, c = r_start + i * dr, c_start + i * dc
                            grid[r][c] = word[i]
                        placed_words.append(word)
                        word_placed = True
                        break # Word placed, move to next word
                if word_placed:
                    break # Break from c_start loop
            if word_placed:
                break # Break from r_start loop

        if not word_placed:
            print(f"Warning: Could not place word '{word}'. It might be too long or the grid too full/constrained.")

    # Fill empty spaces with a fixed character
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] == ' ':
                grid[r][c] = FILL_CHAR

    return grid, placed_words

# --- Word Finding Function ---
# This function does NOT need 'random' and remains largely the same.
def find_word_in_puzzle(puzzle_grid, word_to_find):

    len_word = len(word_to_find)
    word_to_find = word_to_find.upper() # Ensure case consistency

    # DIRECTIONS are copied here for self-containment if we remove global DIRECTIONS
    local_directions = [
        (0, 1),   # Right
        (0, -1),  # Left
        (1, 0),   # Down
        (-1, 0),  # Up
        (1, 1),   # Down-Right
        (1, -1),  # Down-Left
        (-1, 1),  # Up-Right
        (-1, -1)  # Up-Left
    ]

    for r_start in range(GRID_SIZE):
        for c_start in range(GRID_SIZE):
            if puzzle_grid[r_start][c_start] != word_to_find[0]:
                continue

            for dr, dc in local_directions: # Use local_directions here
                coords = []
                is_match = True
                for i in range(len_word):
                    r_curr, c_curr = r_start + i * dr, c_start + i * dc

                    if not (0 <= r_curr < GRID_SIZE and 0 <= c_curr < GRID_SIZE):
                        is_match = False
                        break
                    if puzzle_grid[r_curr][c_curr] != word_to_find[i]:
                        is_match = False
                        break
                    coords.append((r_curr, c_curr))

                if is_match:
                    return True, coords
    return False, None

# --- Main execution block (for non-interactive testing) ---
if __name__ == "__main__":
    # --- Define your words here for testing ---
    # Example words:
    initial_words = ["LEARNING", "SCIENCE", "FUN", "PYTHON", "CODE", "CHALLENGE", "PROGRAM"]

    # Filter and validate words
    valid_words = []
    for word in initial_words:
        word = word.strip().upper()
        if not word.isalpha():
            print(f"Skipping '{word}': Contains non-alphabetic characters.")
            continue
        if len(word) > GRID_SIZE:
            print(f"Skipping '{word}': Word is too long for a {GRID_SIZE}x{GRID_SIZE} grid.")
            continue
        valid_words.append(word)

    if not valid_words:
        print("No valid words provided. Exiting.")
    else:
        puzzle_grid, placed_words_in_puzzle = create_crossword(valid_words)

        if puzzle_grid:
            print("\n--- Generated Word Search Puzzle ---")
            for row in puzzle_grid:
                print(" ".join(row))

            print("\n--- Words attempted to be placed ---")
            for word in sorted(valid_words):
                print(f"- {word}")

            print("\n--- Actual Words found in the puzzle (by internal check) ---")
            verified_found_words = []
            for word in valid_words:
                found, _ = find_word_in_puzzle(puzzle_grid, word)
                if found:
                    verified_found_words.append(word)

            if verified_found_words:
                for word in sorted(verified_found_words):
                    print(word)
            else:
                print("No words were successfully found after verification.")

            # --- Automated check for specific words (using a predefined list) ---
            print("\n Automated Word Search Tests ")
            test_words = ["SCIENCE", "FUN", "CHALLENGE", "HELLO", "WORLD", "PYTHON", "LEARNING"]
            for test_word in test_words:
                found, coords = find_word_in_puzzle(puzzle_grid, test_word)
                if found:
                    print(f"TEST: '{test_word}' found at: {coords}")
                else:
                    print(f"TEST: '{test_word}' NOT found.")
        else:
            print("\nFailed to generate a puzzle.")