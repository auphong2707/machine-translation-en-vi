import re
import pandas as pd

# [TEXT PREPROCESSING]
def normalize_string(string: str) -> str:
    """Normalize a string by converting to lowercase and removing non-letter characters.

    Args:
        string (str): Input string to normalize.

    Returns:
        str: Normalized string.
    """
    string = string.lower().strip()
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+", r" ", string)
    return string.strip()

def preprocess_data_csv(dir: str) -> list:
    """Reads and preprocesses data from a CSV file.
    
    Args:
        dir (str): Path to the CSV file.
        
    Returns:
        list: A list of sentence pairs.
    """
    data = pd.read_csv(dir, encoding='utf-8')
    data = data.iloc[:, :2]
    data = data.applymap(lambda x: normalize_string(x) if isinstance(x, str) else x)
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.split().str.join('_')
    pairs = data.values.tolist()
    return pairs