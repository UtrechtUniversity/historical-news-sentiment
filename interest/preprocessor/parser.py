import json
import lzma
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Union, Dict
import logging

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
    newsletter_metadata: List[Dict] = []


    try:
        tree=et.parse(meta_file_list[0])
        root=tree.getroot()
    except et.ParseError as e:
        logging.error("Failed to parse the xml file:%s", e)

        
    title_values = [element.text for element in root.iter() if element.tag.endswith('title')]
    if len(title_values)>1:
        logging.warning("More than one titles are extracted from metadata.")
    if not title_values:
        logging.warning("No title is extracted.")
        title = None
    else:
        title = title_values[0]

    language_values = [element.text for element in root.iter() if element.tag.endswith('language')]
    if len(language_values)>1:
        logging.warning("More than one language are extracted from metadata.")
    if not language_values:
        logging.warning("No language is extracted.")
        language = None
    else:
        language = language_values[0]

    
    issuenumber_values = [element.text for element in root.iter() if element.tag.endswith('issuenumber')]
    if len(issuenumber_values)>1:
        logging.warning("More than one issuenumbers are extracted from metadata.")
    if not issuenumber_values:
        logging.warning("No issuenumber is extracted.")
        issuenumber = None
    else:
        issuenumber = issuenumber_values[0]


    date_values = [element.text for element in root.iter() if element.tag.endswith('date')]
    if len(date_values)>1:
        logging.warning("More than one dates are extracted from metadata.")
    if not date_values:
        logging.warning("No date is extracted.")
        date = None
    else:
        date = date_values[0]

    identifier_values = [element.text for element in root.iter() if element.tag.endswith('identifier')]
    if len(identifier_values)>1:
        logging.warning("More than one identifiers are extracted from metadata.")
    if not identifier_values:
        logging.warning("No identifier is extracted.")
        identifier = None
    else:
        identifier = identifier_values[0]

    temporal_values = [element.text for element in root.iter() if element.tag.endswith('temporal')]
    if len(temporal_values)>1:
        logging.warning("More than one temporal are extracted from metadata.")
    if not temporal_values:
        logging.warning("No temporal is extracted.")
        temporal = None
    else:
        temporal = temporal_values[0]

    recordRights_values = [element.text for element in root.iter() if element.tag.endswith('recordRights')]
    if len(recordRights_values)>1:
        logging.warning("More than one recordRights are extracted from metadata.")
    if not recordRights_values:
        logging.warning("No recordRights is extracted.")
        recordRights = None
    else:
        recordRights = recordRights_values[0]

    publisher_values = [element.text for element in root.iter() if element.tag.endswith('publisher')]
    if len(publisher_values)>1:
        logging.warning("More than one publisher are extracted from metadata.")
    if not publisher_values:
        logging.warning("No publisher is extracted.")
        publisher = None
    else:
        publisher = publisher_values[0]

    spatial_values = [element.text for element in root.iter() if element.tag.endswith('spatial')]
    if len(spatial_values)>1:
        logging.warning("More than one spatial are extracted from metadata.")
    if not spatial_values:
        logging.warning("No spatial is extracted.")
        spatial_1 = None
        spatial_2 = None
    else:
        spatial_1 = spatial_values[0]
        spatial_2 = spatial_values[1]

    source_values = [element.text for element in root.iter() if element.tag.endswith('source')]
    if len(source_values)>1:
        logging.warning("More than one source are extracted from metadata.")
    if not source_values:
        logging.warning("No source is extracted.")
        source = None
    else:
        source = source_values[1]

    recordIdentifier_values = [element.text for element in root.iter() if element.tag.endswith('recordIdentifier')]
    if len(recordIdentifier_values)>1:
        logging.warning("More than one recordIdentifier are extracted from metadata.")
    if not recordIdentifier_values:
        logging.warning("No recordIdentifier is extracted.")
        recordIdentifier = None
    else:
        recordIdentifier = recordIdentifier_values[0]

    type_values = [element.text for element in root.iter() if element.tag.endswith('type')]
    if len(type_values)>1:
        logging.warning("More than one type are extracted from metadata.")
    if not type_values:
        logging.warning("No type is extracted.")
        type = None
    else:
        type = type_values[0]

    isPartOf_values = [element.text for element in root.iter() if element.tag.endswith('isPartOf')]
    if len(isPartOf_values)>1:
        logging.warning("More than one isPartOf are extracted from metadata.")
    if not isPartOf_values:
        logging.warning("No isPartOf is extracted.")
        isPartOf_1 = None
        isPartOf_2 = None
    else:
        isPartOf_1 = isPartOf_values[0]
        isPartOf_2 = isPartOf_values[1]


    newsletter_metadata.append({
        "title": title,
          "language":language,
            "issue_number":issuenumber,
            "date": date,
            "identifier": identifier,
            "temporal": temporal,
            "recordRights": recordRights,
            "publisher": publisher,
            "spatial_1": spatial_1,
            "spatial_2": spatial_2,
            "source": source,
            "recordIdentifier": recordIdentifier,
            "type": type,
            "isPartOf_1":isPartOf_1,
            "isPartOf_2":isPartOf_2
            })
     
    return newsletter_metadata


    

if __name__ == "__main__":
    # print(parse_raw_article('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100/MMKB12_000002100_00022_text.xml'))
    # print(parse_journal_articles('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100'))
    print(parse_meta_file('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100'))
    
