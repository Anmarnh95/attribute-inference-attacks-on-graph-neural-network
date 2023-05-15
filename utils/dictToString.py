def dictToString(dictionary: dict):
    return_string = ""
    keys = list(dictionary.keys())
    for i in range(len(keys)):
        return_string = return_string + f"{keys[i]}{dictionary[keys[i]]}"
    return return_string
