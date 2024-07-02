import pickle

def print_pickled_data(file_path):
    try:
        # Open the pickle file for reading in binary mode
        with open(file_path, 'rb') as file:
            # Load the data from the pickle file
            data = pickle.load(file)

            # Print the loaded data
            print(data)
    
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error: {e}")

# Replace 'your_file_path.pkl' with the actual path to your pickle file
file_path = '20.pickle'
print_pickled_data(file_path)