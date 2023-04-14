import sys
import gdown


def create_database_dir():
    url = "https://drive.google.com/drive/folders/1BjzJC6voM8KP3OxyD2LXXkvgAoataW1a?usp=share_link"
    gdown.download_folder(url, quiet=False, use_cookies=True)


if __name__ == "__main__":
    if len(sys.argv)  == 1:
        raise Exception("No arguments passed some arguments. Use -h/--help for help")
        
    if sys.argv[1] == "--create_database":
        print("May take a few minutes.")
        print("Downloading Databases Directory...")
        create_database_dir()
    elif sys.argv[1] in ["-h", "--help"]:
        print(
            """
            python dataset.py [-h]                Provides documentation
                              [--help]
                              
                              [--create_database] Creates the "Databases" directory 
                                                  in the root directory with all the datasets
            """
        )
    else:
        raise("Please see the documentation with -h")