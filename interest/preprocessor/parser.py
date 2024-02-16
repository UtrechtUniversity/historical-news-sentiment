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
    
def parse_meta_file(input_dir: Union[Path, str]) -> Dict:
    input_dir = Path(input_dir)
    meta_file_list = list(input_dir.glob("*.didl.xml"))

    tree=et.parse(meta_file_list[0])
    root=tree.getroot()

    title_values = [element.text for element in root.iter() if element.tag.endswith('title')]
    language_values = [element.text for element in root.iter() if element.tag.endswith('language')]
    issuenumber_values = [element.text for element in root.iter() if element.tag.endswith('issuenumber')]
    date_values = [element.text for element in root.iter() if element.tag.endswith('date')]
    identifier_values = [element.text for element in root.iter() if element.tag.endswith('identifier')]

    print(title_values[0], '*')
    print(language_values[0], '*')
    print(issuenumber_values[0], '*')
    print(date_values[0], '*')
    print(identifier_values[0], '*')



    



    

    # for item in root.findall('.//{urn:mpeg:mpeg21:2002:02-DIDL-NS}Item'):
    #     for x in item.iter():
    #         for t in x.findall('{http://purl.org/dc/elements/1.1/}title'):
    #             title = t.text
    #         for l in x.findall('./{http://purl.org/dc/elements/1.1/}language'):
    #             language = l.text
    #         for issuenumber in x.findall('./{http://krait.kb.nl/coop/tel/handbook/telterms.html}issuenumber'):
    #             issue_number = issuenumber.text
    #         for d in x.findall('./{http://purl.org/dc/elements/1.1/}date'):
    #             date = d.text
    #         for i in x.findall('./{http://purl.org/dc/elements/1.1/}identifier'):
    #             identifier = i.text
            





    

if __name__ == "__main__":
    # print(parse_raw_article('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100/MMKB12_000002100_00022_text.xml'))
    # print(parse_journal_articles('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100'))
    parse_meta_file('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100')
    
