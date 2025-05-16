import random
total=0
trigger=1
coverage_file=""
total_coverage=0.0
import matplotlib.pyplot as plt
# Add these after your other global variables
coverage_history = []
generation_history = []
import csv

def generate_best_evolved_test_cases_csv():
    """
    Generates a CSV file with the best-evolved test cases and their categories.
    This should be called right before the program ends, after all generations.
    """
    print("Generating CSV file with best-evolved test cases...")
    
    # Determine which file to read from
    input_file = "mutation.txt" if total_coverage > 0 else "iteration_2.txt"
    output_file = "best_evolved_test_cases.csv"
    
    best_test_cases = []
    
    # Read the latest iteration's test cases
    try:
        with open(input_file, "r") as file:
            lines = file.readlines()
            
        for line in lines:
            if " - Fitness: " in line:
                parts = line.strip().split(" - Fitness: ")
                date = parts[0]
                fitness = float(parts[1])
                best_test_cases.append((date, fitness))
        
        # Sort by fitness in descending order
        best_test_cases.sort(key=lambda x: x[1], reverse=True)
        
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found. Cannot generate CSV.")
        return
    
    # Categorize test cases
    categorized_test_cases = []
    for date, fitness in best_test_cases:
        try:
            day, month, year = map(int, date.split('/'))
            
            # Determine categories
            categories = []
            
            # Category 1: Month type
            if month == 2:
                month_type = "February"
            elif month in [4, 6, 9, 11]:
                month_type = "30-day month"
            elif month in [1, 3, 5, 7, 8, 10, 12]:
                month_type = "31-day month"
            else:
                month_type = "Invalid month"
            
            # Category 2: Day edge cases
            if month == 2 and day == 29:
                day_category = "Leap day"
            elif month == 2 and day == 28:
                day_category = "Last day of February (non-leap)"
            elif month in [4, 6, 9, 11] and day == 30:
                day_category = "Last day of 30-day month"
            elif month in [1, 3, 5, 7, 8, 10, 12] and day == 31:
                day_category = "Last day of 31-day month"
            else:
                day_category = "Regular day"
            
            # Category 3: Year edge cases
            if year == 0: 
                year_category = "Minimum year"
            elif year == 9999:
                year_category = "Maximum year"
            elif (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                year_category = "Leap year"
            else:
                year_category = "Regular year"
                
            categorized_test_cases.append({
                'test_case': date,
                'fitness': fitness,
                'month_type': month_type,
                'day_category': day_category,
                'year_category': year_category
            })
            
        except Exception as e:
            print(f"❌ Error categorizing test case {date}: {str(e)}")
            # Add with minimal categorization
            categorized_test_cases.append({
                'test_case': date,
                'fitness': fitness,
                'month_type': "Unknown",
                'day_category': "Unknown",
                'year_category': "Unknown"
            })
    
    # Write to CSV
    try:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['test_case', 'fitness', 'month_type', 'day_category', 'year_category']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for test_case in categorized_test_cases:
                writer.writerow(test_case)
        
        print(f"✅ CSV file successfully generated: {output_file}")
        
    except Exception as e:
        print(f"❌ Error writing to CSV file: {str(e)}")

def generate_coverage_graph(coverage_values, generation_numbers):
    """
    Generates a line graph showing coverage improvement over generations.
    
    Parameters:
    - coverage_values: List of coverage percentages
    - generation_numbers: List of generation numbers
    """
    plt.figure(figsize=(10, 6))
    plt.plot(generation_numbers, coverage_values, 'bo-', linewidth=2, markersize=8)
    plt.title('GA Coverage Improvement Over Generations', fontsize=16)
    plt.xlabel('Generation Number', fontsize=14)
    plt.ylabel('Coverage (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a horizontal line at 95% to show the target coverage
    plt.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='Target Coverage (95%)')
    
    # Annotate the final coverage value
    final_gen = generation_numbers[-1]
    final_coverage = coverage_values[-1]
    plt.annotate(f'{final_coverage:.2f}%', 
                xy=(final_gen, final_coverage),
                xytext=(final_gen-5, final_coverage+5),
                arrowprops=dict(arrowstyle='->'))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the graph
    plt.savefig('coverage_improvement.png')
    plt.close()
    
    print("✅ Coverage improvement graph saved as 'coverage_improvement.png'")

# ✅ Generate a random test case (Ensures date format remains consistent)
def generate_random_date():
    day = random.randint(1, 40)   # Allow days > 31 for later invalid cases
    month = random.randint(1, 20) # Allow months > 12 for later invalid cases
    year = random.randint(0, 11000)  # Allow years > 9999 for later invalid cases
    return f"{day:02d}/{month:02d}/{year:04d}"

# ✅ Generate and save 50 test cases
def generate_test_cases():
    test_cases = [generate_random_date() for _ in range(100)]
    with open("iteration_1.txt", "w") as file:
        for test in test_cases:
            file.write(test + "\n")
    print("✅ 50 test cases saved to iteration_1.txt!")

# Call the function to generate test cases
generate_test_cases()
import re

# ✅ Read test cases from file (Returns list of strings)
def read_test_cases():
    with open("iteration_1.txt", "r") as file:
        return [line.strip() for line in file.readlines()]

def single_point_crossover_pair(parent1, parent2):
    """
    Performs single-point crossover correctly and returns two offspring.
    Ensures that one offspring follows the image example of crossover.
    """
    # Add a maximum number of attempts to prevent infinite loops
    max_attempts = 100
    attempts = 0
    
    while attempts < max_attempts:  # Prevent infinite loop
        attempts += 1
        day1, month1, year1 = map(int, parent1.split('/'))
        day2, month2, year2 = map(int, parent2.split('/'))

        crossover_point = random.randint(0, 2)  # 0 = day, 1 = month, 2 = year

        if crossover_point == 0:  # Swap day
            child1 = f"{day2:02d}/{month1:02d}/{year1:04d}"
            child2 = f"{day1:02d}/{month2:02d}/{year2:04d}"
        elif crossover_point == 1:  # Swap month
            child1 = f"{day1:02d}/{month2:02d}/{year1:04d}"
            child2 = f"{day2:02d}/{month1:02d}/{year2:04d}"
        else:  # Swap year
            child1 = f"{day1:02d}/{month1:02d}/{year2:04d}"
            child2 = f"{day2:02d}/{month2:02d}/{year1:04d}"

        # Validate the generated offspring
        if is_valid_date(child1) and is_valid_date(child2):
            return child1, child2  # ✅ If valid, return both offspring
    
    # If we can't find valid offspring after max attempts, return modified versions of parents
    # This ensures we always return something
    return parent1, parent2
# Function to check if a date is valid
def is_valid_date(date_str):
    """
    Validates a date string in DD/MM/YYYY format.
    Returns True if valid, False if invalid.
    """
    match = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", date_str)
    if not match:
        return False  # Wrong format
    
    day, month, year = map(int, match.groups())

    # Validate month
    if not (1 <= month <= 12):
        return False

    # Validate day
    if not (1 <= day <= 31):
        return False

    # Validate year
    if not (0 <= year <= 9999):  # Ensure year is within bounds
        return False

    # Days in 30-day months
    if month in [4, 6, 9, 11] and day > 30:
        return False

    # Check February
    if month == 2:
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        max_day = 29 if is_leap else 28
        if day > max_day:
            return False

    return True

def calculate_fitness(valid_dates):
    fitness_scores = {}
    unique_months, unique_days, unique_years = set(), set(), set()

    for date in valid_dates:
        date = date.strip()
        if not is_valid_date(date):
            print(f"⚠️ Skipping invalid date format: {date}")
            continue  

        try:
            day, month, year = map(int, date.split('/'))
        except ValueError:
            print(f"❌ Error: Could not parse date: {date}")
            continue  

        fitness = 0
        redundant_cases = 0  

        if month not in unique_months:
            fitness += 1
            unique_months.add(month)
        else:
            redundant_cases += 1  

        if day not in unique_days:
            fitness += 1
            unique_days.add(day)
        else:
            redundant_cases += 1  

        if year not in unique_years:
            fitness += 1
            unique_years.add(year)
        else:
            redundant_cases += 1  

        special_case = False
        if (month == 2 and day in [28, 29]) or (month in [4, 6, 9, 11] and day == 30) or \
           (month in [1, 3, 5, 7, 8, 10, 12] and day == 31) or (year in [0, 9999]):
            fitness += 15
            special_case = True

        if special_case:
            redundant_cases = 0  

        fitness_value = fitness if redundant_cases == 0 else fitness / (1 + redundant_cases)
        fitness_scores[date] = fitness_value

    return fitness_scores

# ✅ Validate and store results
def validate_and_store_results():
    test_cases = read_test_cases()
    valid_dates, invalid_dates = [], []

    for date in test_cases:
        if is_valid_date(date):
            valid_dates.append(date)
        else:
            invalid_dates.append(date)

    valid_dates = list(map(str, valid_dates))  # Ensures no tuples
    fitness_scores = calculate_fitness(valid_dates)

    with open("iteration_1_validation.txt", "w") as file:
        file.write("Valid Dates (With Fitness Scores):\n")
        for date, fitness in fitness_scores.items():
            file.write(f"{date} - Fitness: {fitness:.2f}\n")
        file.write("\nInvalid Dates:\n")
        for date in invalid_dates:
            file.write(f"{date}\n")

    print("✅ Validation complete! Results saved to iteration_1_validation.txt.")

# Run the validation and fitness calculation
validate_and_store_results()
def calculate_coverage():
    """
    Reads the top 10 fitness values from iteration_1_validation.txt,
    calculates coverage, and stores the results in iteration_coverage.txt.
    """
    validation_file = "iteration_1_validation.txt"
    coverage_file = "iteration_coverage.txt"
    
    test_cases_with_fitness = []

    # Read and extract test cases with fitness values
    with open(validation_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        if " - Fitness: " in line:  # Identify valid test cases with fitness
            parts = line.strip().split(" - Fitness: ")
            date = parts[0]
            fitness = float(parts[1])
            test_cases_with_fitness.append((date, fitness))

    # Sort test cases by highest fitness
    test_cases_with_fitness.sort(key=lambda x: x[1], reverse=True)

    # Select top 10 test cases
    top_10 = test_cases_with_fitness[:10]

    # Compute total fitness sum and coverage
    total_fitness_sum = sum(fitness for _, fitness in top_10)
    coverage = (total_fitness_sum / 155) * 100  # Maximum possible coverage is 180

    # Write results to iteration_coverage.txt
    with open(coverage_file, "w") as file:
        file.write("Top 10 Test Cases (With Fitness Scores):\n")
        for date, fitness in top_10:
            file.write(f"{date} - Fitness: {fitness:.2f}\n")
        
        file.write("\nCoverage Achieved: {:.2f}%\n".format(coverage))

    print("✅ Coverage calculation complete! Results saved to iteration_coverage.txt.")

# Run the coverage calculation function
calculate_coverage()
def can_get_15_point_boost(day, month, year):
    """
    Returns True if the given (day, month, year) combination qualifies for a +15 fitness boost.
    """
    if (month == 2 and day == 29) or \
       (month == 2 and day == 28) or \
       (month in [4, 6, 9, 11] and day == 30) or \
       (month in [1, 3, 5, 7, 8, 10, 12] and day == 31) or \
       (year == 0 or year == 9999):
        return True
    return False
import random
def get_top_5_parents(coverage_file):
    """
    Reads the given coverage file (either iteration_coverage.txt or mutation.txt),
    extracts test cases with their fitness values, and returns the top 5 parents.
    """
    top_parents = []

    try:
        # Read the file and extract test cases with fitness values
        with open(coverage_file, "r") as file:
            lines = file.readlines()

        for line in lines:
            if " - Fitness: " in line:
                parts = line.strip().split(" - Fitness: ")
                date = parts[0]
                fitness = float(parts[1])

                # ✅ Store date and fitness as a tuple
                top_parents.append((date, fitness))

        # ✅ Special handling for mutation.txt (unordered dates)
        if "mutation" in coverage_file.lower():
            # Since mutation.txt is unordered, explicitly sort by fitness
            top_parents.sort(key=lambda x: x[1], reverse=True)

        # ✅ Select the top 5 test cases
        return top_parents[:5] if len(top_parents) >= 5 else top_parents

    except FileNotFoundError:
        print(f"❌ Error: {coverage_file} not found.")
        return []
    except Exception as e:
        print(f"❌ Unexpected error reading {coverage_file}: {str(e)}")
        return []

import random
def heuristic_crossover():
    global coverage_file, trigger

    if trigger == 1:
        coverage_file = "iteration_coverage.txt"
        trigger = 0
    else:
        coverage_file = "mutation.txt"

    parents_with_fitness = get_top_5_parents(coverage_file)
    parents = [p[0] for p in parents_with_fitness]

    if len(parents) < 5:
        print("❌ Not enough top fitness test cases found. Exiting.")
        exit()

    offsprings = []
    parent_usage = {parent: 0 for parent in parents}
    peak_parent = max(parents_with_fitness, key=lambda x: x[1])[0]

    # Try every parent against every other parent to maximize fitness
    for i in range(5):
        if len(offsprings) >= 5:
            break
            
        parent1 = parents[i]

        # Ensure peak fitness parent is used **exactly twice**
        if parent1 == peak_parent and parent_usage[parent1] >= 2:
            continue  # Skip if already used twice

        # Ensure other parents are used **only once**
        if parent1 != peak_parent and parent_usage[parent1] >= 1:
            continue  # Skip if already used once

        for j in range(5):
            if i == j or len(offsprings) >= 5:
                continue  # Skip pairing with itself or if we have enough offspring

            parent2 = parents[j]

            # Ensure parent2 also follows the same constraints
            if parent2 == peak_parent and parent_usage[parent2] >= 2:
                continue
            if parent2 != peak_parent and parent_usage[parent2] >= 1:
                continue

            # Try different crossover points
            day1, month1, year1 = map(int, parent1.split('/'))
            day2, month2, year2 = map(int, parent2.split('/'))
            
            best_child_1 = None
            best_child_2 = None
            best_fitness_1 = -1
            
            for crossover_point in range(3):  # 0 = day, 1 = month, 2 = year
                if crossover_point == 0:  # Swap day
                    new_day_1, new_month_1, new_year_1 = day2, month1, year1
                    new_day_2, new_month_2, new_year_2 = day1, month2, year2
                elif crossover_point == 1:  # Swap month
                    new_day_1, new_month_1, new_year_1 = day1, month2, year1
                    new_day_2, new_month_2, new_year_2 = day2, month1, year2
                else:  # Swap year
                    new_day_1, new_month_1, new_year_1 = day1, month1, year2
                    new_day_2, new_month_2, new_year_2 = day2, month2, year1

                child1 = f"{new_day_1:02d}/{new_month_1:02d}/{new_year_1:04d}"
                child2 = f"{new_day_2:02d}/{new_month_2:02d}/{new_year_2:04d}"

                # Validate both children
                if is_valid_date(child1) and is_valid_date(child2):
                    # Check if child1 qualifies for the +15 boost
                    if can_get_15_point_boost(new_day_1, new_month_1, new_year_1):
                        child_fitness_1 = 15  
                    else:
                        child_fitness_1 = calculate_fitness([child1])[child1]

                    # Check if this is the best fitness found so far
                    if child_fitness_1 > best_fitness_1 and child1 != parent1 and child1 != parent2:
                        best_fitness_1 = child_fitness_1
                        best_child_1 = child1
                        best_child_2 = child2

            # If we found good children, add them and update parent usage
            if best_child_1 and best_child_2:
                # Only increment parent usage if we're actually adding children
                parent_usage[parent1] += 1
                parent_usage[parent2] += 1
                
                # Add the children to the offspring list
                if len(offsprings) < 5:
                    offsprings.append(best_child_1)
                if len(offsprings) < 5:
                    offsprings.append(best_child_2)
                
                # If we have enough offspring, we're done
                if len(offsprings) >= 5:
                    break
    
    # If we couldn't find 5 offspring with heuristic crossover, use fallback approach
    if len(offsprings) < 5:
        # Safety check for available_parents to prevent sampling errors
        available_parents = []
        for parent in parents:
            if (parent == peak_parent and parent_usage[parent] < 2) or \
               (parent != peak_parent and parent_usage[parent] < 1):
                available_parents.append(parent)
        
        # If we don't have enough available parents, use any parents
        if len(available_parents) < 2:
            available_parents = parents
        
        # Generate additional offspring with a fallback timeout
        max_fallback_attempts = 20  # Prevent infinite loops
        fallback_attempts = 0
        
        while len(offsprings) < 5 and fallback_attempts < max_fallback_attempts:
            fallback_attempts += 1
            
            if len(available_parents) >= 2:
                # Choose two different parents safely
                idx1 = random.randint(0, len(available_parents) - 1)
                idx2 = random.randint(0, len(available_parents) - 1)
                while idx2 == idx1 and len(available_parents) > 1:
                    idx2 = random.randint(0, len(available_parents) - 1)
                    
                parent1, parent2 = available_parents[idx1], available_parents[idx2]
            else:
                # Fallback if we somehow don't have enough parents
                parent1 = available_parents[0]
                parent2 = parents[random.randint(0, len(parents) - 1)]
            
            # Generate offspring
            child1, child2 = single_point_crossover_pair(parent1, parent2)
            
            # Add unique offspring
            if child1 not in offsprings and len(offsprings) < 5:
                offsprings.append(child1)
            
            if child2 not in offsprings and len(offsprings) < 5:
                offsprings.append(child2)
        
        # Last resort: If we still don't have 5 offspring, just use parents again
        while len(offsprings) < 5:
            # Add a parent as an offspring (not ideal but prevents hanging)
            parent_idx = len(offsprings) % len(parents)
            offsprings.append(parents[parent_idx])
        
        # Ensure we have exactly 5 offspring
        offsprings = offsprings[:5]
    
    return parents_with_fitness, offsprings



def generate_offspring():
    parents = get_top_5_parents(coverage_file)  # Get the top 5 fitness-level dates
    if len(parents) < 5:
        print("❌ Not enough top fitness test cases found. Exiting.")
        exit()

    offsprings = []

    # Perform single-point crossover to generate 4 offspring
    for i in range(4):
        parent1, _ = parents[i]  
        parent2, _ = parents[i+1]  
        child1, child2 = single_point_crossover_pair(parent1, parent2)
        offsprings.append(child1)
        offsprings.append(child2)


    # Use the best fitness-level date with the last one to get the 5th offspring
    best_parent = parents[0][0]
    last_parent = parents[4][0]
    final_offspring = single_point_crossover_pair(best_parent, last_parent)
    offsprings.append(final_offspring)

    return parents, offsprings

def save_iteration_2():
    global total_coverage, total
    global total_coverage, total, coverage_history, generation_history
    
    try:
        generation_count = 0
        
        # Set a timeout for the heuristic_crossover function
        import time
        start_time = time.time()
        timeout = 10  # seconds
        
        parents_with_fitness = []
        offsprings = []
        
        # Try to get the best possible offspring using heuristic crossover
        try:
            parents_with_fitness, offsprings = heuristic_crossover()
            if time.time() - start_time > timeout:
                raise TimeoutError("Heuristic crossover took too long")
        except Exception as e:
            print(f"⚠️ Error in heuristic crossover: {str(e)}")
            # Fallback to simple approach
            parents_with_fitness = get_top_5_parents(coverage_file)
            offsprings = []
        
        generation_count = len(offsprings)
        
        # Ensure offsprings are only strings
        offsprings = [str(offspring) for offspring in offsprings if isinstance(offspring, str)][:5]
        
        # Ensure we have exactly 5 parents
        parent_dates = [str(parent[0]) for parent in parents_with_fitness if isinstance(parent, tuple)][:5]
        
        # Emergency fallback if we don't have 5 parents
        while len(parent_dates) < 5:
            parent_dates.append("01/01/2000")  # Add a default date
        
        # Generate additional offspring if needed
        while len(offsprings) < 5:
            # Choose two random parents
            parent1 = random.choice(parent_dates)
            parent2 = random.choice(parent_dates)
            
            # Generate new offspring
            try:
                child1, child2 = single_point_crossover_pair(parent1, parent2)
                
                # Add unique offspring
                if child1 not in offsprings and len(offsprings) < 5:
                    offsprings.append(child1)
                    generation_count += 1
                    
                if child2 not in offsprings and len(offsprings) < 5:
                    offsprings.append(child2)
                    generation_count += 1
            except Exception as e:
                print(f"⚠️ Error generating offspring: {str(e)}")
                # Add a fallback date
                offsprings.append("01/01/2001")
        
        # Ensure we have exactly 5 offspring
        offsprings = offsprings[:5]
        
        # Combine parent dates and offspring dates
        all_dates = parent_dates + offsprings
        
        # Calculate fitness for all dates
        all_fitness = calculate_fitness(all_dates)
        
        # Calculate maximum possible fitness
        max_fitness = 0
        for date in all_dates:
            day, month, year = map(int, date.split('/'))
            if (month == 2 and day in [28, 29]) or (month in [4, 6, 9, 11] and day == 30) or \
               (month in [1, 3, 5, 7, 8, 10, 12] and day == 31) or (year in [0, 9999]):
                max_fitness += 18  # Base 3 + special case 15
            else:
                max_fitness += 3   # Base value (1 for each component)
        
        # If all dates are special cases, max is 180 (10 dates * 18)
        # But if not, we need to calculate properly
        if max_fitness == 0:
            max_fitness = 155  # Fallback to your original value
        
        # Write to file
        with open("iteration_2.txt", "w") as file:
            file.write("Top 5 Parents & Offspring (With Fitness Scores):\n")
            
            total_fitness = 0
            for date in all_dates:
                fitness_value = all_fitness.get(date, 0)
                total_fitness += fitness_value
                file.write(f"{date} - Fitness: {fitness_value:.2f}\n")
            
            # Calculate coverage using actual fitness values
            coverage = (total_fitness / 155) * 100  # Using 155 as the reference value
            total_coverage = coverage
            
            file.write("\nCoverage Achieved: {:.2f}%\n".format(coverage))
            file.write("Number of Generations Used: {}\n".format(generation_count))
            total += generation_count
        
        print("✅ Iteration 2 complete! Results saved to iteration_2.txt.")
        print("✅ Coverage Achieved: {:.2f}%".format(coverage))
        print("✅ Number of Generations Used:", generation_count)
            # At the end of the function, after calculating coverage, add:
        coverage_history.append(coverage)
        generation_history.append(total)
    
    except Exception as e:
        # Global error handler to prevent the program from hanging
        print(f"❌ Error in save_iteration_2: {str(e)}")
        
        # Create a minimal valid file to allow the program to continue
        with open("iteration_2.txt", "w") as file:
            file.write("Top 5 Parents & Offspring (With Fitness Scores):\n")
            for i in range(10):
                file.write(f"01/0{i+1}/2000 - Fitness: 1.00\n")
            file.write("\nCoverage Achieved: 10.00%\n")
            file.write("Number of Generations Used: 5\n")
        
        # Set default values to allow the program to continue
        total_coverage = 10.0
        total += 5
        
        print("✅ Iteration 2 complete! (Recovery mode) Results saved to iteration_2.txt.")
        print("✅ Coverage Achieved: 10.00%")
        print("✅ Number of Generations Used: 5")
def probability_game():
    """
    Determines if a mutation occurs based on a 15% probability.
    Returns True if mutation happens, otherwise False.
    """
    return random.random() < 0.15  # 15% probability
def mutation_heuristic(date):
    if not is_valid_date(date):
        return None  

    try:
        day, month, year = map(int, date.split('/'))
    except ValueError:
        return None  

    for delta in [-3, -2, -1, 1, 2, 3]:
        new_day = day + delta
        new_date = f"{new_day:02d}/{month:02d}/{year:04d}"
        if is_valid_date(new_date):
            return new_date

    for delta in [-1, 1]:
        new_month = month + delta
        new_date = f"{day:02d}/{new_month:02d}/{year:04d}"
        if is_valid_date(new_date):
            return new_date

    for delta in range(-100, 101):
        new_year = year + delta
        new_date = f"{day:02d}/{month:02d}/{new_year:04d}"
        if is_valid_date(new_date):
            return new_date

    return None  
def perform_valid_mutation(date):
    """
    If the mutation heuristic fails, this function applies a random valid mutation.
    - Randomly modifies day, month, or year while ensuring validity.
    """
    day, month, year = map(int, date.split('/'))
    
    while True:
        mutation_type = random.choice(["day", "month", "year"])
        
        if mutation_type == "day":
        # Change day by at most ±3 within valid bounds
          new_day = max(1, min(31, day + random.choice([-3, -2, -1, 1, 2, 3])))
          mutated_date = f"{new_day:02d}/{month:02d}/{year:04d}"

        elif mutation_type == "month":
        # Change month by at most ±1 within valid bounds
           new_month = max(1, min(12, month + random.choice([-1, 1])))
           mutated_date = f"{day:02d}/{new_month:02d}/{year:04d}"

        else:  # "year"
        # Change year by any value between -100 and +100
          new_year = max(0, min(9999, year + random.randint(-100, 100)))
          mutated_date = f"{day:02d}/{month:02d}/{new_year:04d}"
        if is_valid_date(mutated_date):
           return mutated_date  # Ensure only valid dates are returned
        
def mutate_dates():
    """
    Reads dates from iteration_2.txt, applies mutation probability check,
    calls mutation functions, calculates fitness for mutated dates,
    and saves the results to mutation.txt.
    """
    input_file = "iteration_2.txt"
    output_file = "mutation.txt"
    
    dates = []

    # ✅ Read iteration_2.txt and extract the top 10 dates along with their fitness values
    with open(input_file, "r") as file:
        lines = file.readlines()  # Ensure we read the file inside the 'with open' block

    for line in lines:
        if " - Fitness: " in line:  # ✅ Ensure this line contains a date and fitness value
            parts = line.strip().split(" - Fitness: ")
            date = parts[0]
            fitness = float(parts[1])  # Convert fitness value to float
            dates.append((date, fitness))  # ✅ Store as a tuple (date, fitness)

    # ✅ Limit to 10 dates (if available)
    dates = dates[:10]

    if not dates:  # If dates list is empty, print error and return
        print("❌ No valid dates found in iteration_2.txt for mutation.")
        return

    mutated_dates = []
    generations_used = 0  # Track number of actual date mutations

    for date, fitness in dates:  # ✅ Now we have both date and fitness value
        if fitness >= 15:  # ✅ Skip mutation for dates with fitness ≥ 15
            mutated_dates.append(date)  # Keep date unchanged
            continue  # Move to the next date

        if probability_game():  # ✅ 15% chance to mutate
            mutated_date = mutation_heuristic(date)

            if not mutated_date:  # If no 15-pointer mutation found, do a valid mutation
                mutated_date = perform_valid_mutation(date)

            mutated_dates.append(mutated_date)

            # ✅ **Increment generations only if the mutated date is different**
            if mutated_date != date:
                generations_used += 1  # ✅ Count mutation as a new generation

        else:
            mutated_dates.append(date)  # ✅ Keep date unchanged

    if not mutated_dates:  # ✅ Check if mutation produced any changes
        print("❌ No mutations occurred. Skipping mutation.txt writing.")
        return

    # ✅ Calculate fitness for mutated dates
    mutated_fitness = calculate_fitness(mutated_dates)

    with open(output_file, "w") as file:
        file.write("Mutated Dates (With Fitness Scores):\n")
    
        total_fitness = 0  # ✅ Track total fitness for coverage calculation
        for date in mutated_dates:
            fitness_value = mutated_fitness.get(date, 0)  # Get fitness, default 0 if not found
            total_fitness += fitness_value  # ✅ Add to total fitness sum
            file.write(f"{date} - Fitness: {fitness_value:.2f}\n")

        # ✅ Calculate coverage (max possible coverage = 180)
        coverage = (total_fitness / 155) * 100 if total_fitness > 0 else 0
        global total_coverage
        global total

        file.write("\nCoverage Achieved: {:.2f}%\n".format(coverage))
        total_coverage=coverage  # ✅ Write coverage
        file.write("Total Generations Used for Mutation: {}\n".format(generations_used))  # ✅ Write generations used
        total=total+generations_used
        coverage_history.append(coverage)
        generation_history.append(total)
    print("✅ Mutation process complete! Results saved to mutation.txt.")
    print("✅ Coverage Achieved: {:.2f}%".format(coverage))  # ✅ Also print coverage in terminal

c1=1
while(c1==1):
  user_input = input("Press 1 to continue, or any other key to exit: ")
  # Check user input
  if user_input == "1":
    print("✅ Proceeding with the program...")
    
    print("✅ Proceeding with Iteration 2...")
    save_iteration_2()
    mutate_dates()
    print ("Total generations used is : ",total)
    if total_coverage>95.00:
        generate_best_evolved_test_cases_csv()
        break
    if total > 100:
      generate_best_evolved_test_cases_csv()
      break
    
  else:
    print("❌ Exiting the program.")
    break

# After the loop ends, call the function to generate the graph
if len(coverage_history) > 0:
    generate_coverage_graph(coverage_history, generation_history)
    print("Program completed successfully!")
    generate_best_evolved_test_cases_csv()
else:
    print("No coverage data was collected.")
    generate_best_evolved_test_cases_csv()

