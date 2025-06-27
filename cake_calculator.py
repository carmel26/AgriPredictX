def cake_calculator(flour, sugar):
    # initial values for ingredients needed to make one cake
    flour_needed = 100   
    sugar_needed = 50    
    cake_count = 0      

    # Loop to calculate how many cakes can be made with the available ingredients
    while True:
        # Check if we have enough ingredients for another cake
        if flour < flour_needed or sugar < sugar_needed:
            break  # Stop if we can't make any more cakes

        # If we have enough, make a cake and update ingredients
        # reduce flour and sugar by the amounts needed for one cake and increment the cake count
        flour -= flour_needed
        sugar -= sugar_needed
        cake_count += 1

    flour_left = flour
    sugar_left = sugar

    return [cake_count, flour_left, sugar_left]

# Main execution block
if __name__ == "__main__":
    print("Welcome to the Cake Calculator!")
    print("Enter the available ingredients to see how many cakes you can make.")

#    check if the user input is valid and handle exceptions
    try:
        # Get input from the user
        available_flour_str = input("Enter the available amount of flour (units): ")
        available_sugar_str = input("Enter the available amount of sugar (units): ")

        # Convert input to integers
        available_flour = int(available_flour_str)
        available_sugar = int(available_sugar_str)

        # Validate input (ensure positive values as per problem description "integer larger 0")
        if available_flour < 0 or available_sugar < 0:
            print("Error: Flour and sugar amounts must be non-negative. Please enter valid numbers.")
        else:
            # Call the cake_calculator function
            result = cake_calculator(available_flour, available_sugar)

            # Display the results
            print("\n--- Calculation Results ---")
            print(f"You started with: {available_flour} units of flour and {available_sugar} units of sugar.")
            print(f"Number of cakes you can make: {result[0]}")
            print(f"Leftover flour: {result[1]} units")
            print(f"Leftover sugar: {result[2]} units")

    except ValueError:
        print("Invalid input. Please enter whole numbers for flour and sugar.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")