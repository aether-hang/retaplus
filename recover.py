import sys
from retapp.cli import main

if __name__ == "__main__":
    main(["recover", *sys.argv[1:]])
