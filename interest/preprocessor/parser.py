import json
import lzma
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Union, Dict, Optional
import logging

import xml.etree.cElementTree as et


class NewsletterFile:
    """ Class for parsing xml files to json """

    def __init__(
            self,
            input_dir: Union[Path, str], 
            output_dir: Union[Path, str]
            ):
             
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)


    def parse_all_articles(self) -> Dict:
        
        file_list = list(self.input_dir.glob("*.xml"))
        # List of meta files
        meta_file_list = list(self.input_dir.glob("*.didl.xml"))
        # List of xml files excluded meta file
        article_file_list = [item for item in file_list if item not in meta_file_list]

        articles: List[Dict] = []


        for file in article_file_list:
            article = self._parse_raw_article(file)
            articles.append(article)

        newsletter_metadata= self._parse_meta_file(meta_file_list[0])

        news_dict = {"newsletter_metadata": newsletter_metadata, "articles": articles}
        return news_dict

    def _parse_raw_article(self, article_fp: Union[Path, str]) -> Dict:
        """Parse a raw article file into a structured list

        Arguments
        ---------
        article_input_fp: Union[Path, str]
        Input file to process.

        Returns
        --------
        articles: List[Dict]
        A list of dictionaries, where each item is for one article and includes
        the title and the body of article.
        
        """
        try:
            tree = et.parse(article_fp)
            root = tree.getroot()
        except et.ParseError as e:
            logging.error("Failed to parse the article file:%s", e)

        title_values = [element.text for element in root.iter() if element.tag.endswith('title')]
        if len(title_values)>1:
            logging.warning("More than one titles are extracted for the article.")
        if not title_values:
            logging.warning("No title is extracted for the article.")
            title = None
        else:
            title = title_values[0]

        body_values = [element.text for element in root.iter() if element.tag.endswith('p')]
        if not body_values:
            logging.warning("No body is extracted.")
            body = None
        if len(body_values)>1:
            logging.warning("There are more than on paragraphs in the article.")
            body = ' '.join(body_values)
        else:
            body = body_values[0]

        return {"title": title, "body":body}
        

    def _parse_meta_file(self, meta_fp: Union[Path, str]) -> Dict:

        newsletter_metadata: List[Dict] = []


        try:
            tree=et.parse(meta_fp)
            root=tree.getroot()
        except et.ParseError as e:
            logging.error("Failed to parse the meta file:%s", e)

            
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
    x = NewsletterFile(input_dir = '../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100', output_dir=' ')
    print(x.parse_all_articles())
    # print(x.input_dir)

    # # print(parse_raw_article('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100/MMKB12_000002100_00022_text.xml'))
    # # print(parse_journal_articles('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100'))
    # # print(parse_meta_file('../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100'))
    # x = parse_all_articles("../../data/news/2022_harvest_KRANTEN/00/KRANTEN_KBPERS01_000002100")
    # print(x)

    
