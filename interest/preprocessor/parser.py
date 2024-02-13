import json
import lzma
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Union, Dict

import xml.etree.cElementTree as et

def parse_raw_article(article_input_fp: Union[Path, str]) -> Dict:
    """Parse a raw article file into a structured list

    Arguments
    ---------
    article_input_fp:
    Input file to process.

    Returns
    --------
    articles: List[Dict]
    A list of dictionaries, where each item is for one article and includes
    the title and the body of article.
     
    """
    if article_input_fp !=None:
        tree = et.parse(article_input_fp)
        root = tree.getroot()
        for title_item in root.findall('./title'):
            title = title_item.text
        for article_item in root.findall('./p'):
            body = article_item.text

        return title, body


def parse_journal_articles(input_dir: Union[Path, str]) -> Dict:
     input_dir = Path(input_dir)
     file_list = list(input_dir.glob("*.xml"))
     meta_file_list = list(input_dir.glob("*.didl.xml"))
     file_list = [item for item in file_list if item not in meta_file_list]
     articles: List[Dict] = []
     for file in file_list:
         title, body = parse_raw_article(file)
         articles.append({"title": title, "body":body})
     return articles
    
    


    

if __name__ == "__main__":
    # print(parse_raw_article('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100/MMKB12_000002100_00022_text.xml'))
    print(parse_journal_articles('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100'))
    
