from sys import exit

if __name__ == '__main__':
    try:
        pass
    except FileNotFoundError as e:
        print("File loading failed: " + e.filename)
        exit(1)
