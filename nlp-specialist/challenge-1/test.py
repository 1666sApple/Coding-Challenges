from challenge1 import extractText

# For self testing 
def main():
    print("Enter 'q' to exit")
    print("\n" + "-"*50 + "\n")

    while True:
        user_input = input("Enter your text: ")
        if user_input.lower() == 'q':
            break
        
        text = extractText(user_input)
        
        print("\nResults:")
        for category, values in text.items():
            if values:
                print(f"\n{category.replace('_', ' ').capitalize()}:")
                for value in values:
                    print(f"  Value: {value['value']}")
                    print(f"  Position: {value['start']} to {value['end']}")
            else:
                print(f"\n{category.replace('_', ' ').capitalize()}: None found")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
