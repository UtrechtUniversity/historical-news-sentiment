
import os
import tarfile
import gzip
import json
import xml.etree.ElementTree as ET
from typing import Dict, Union
import logging


class XMLExtractor:
    def __init__(self, root_dir: str, output_dir: str):
        self.root_dir = root_dir
        self.output_dir = output_dir

    def extract_xml_string(self) -> None:
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if not folder_name.isdigit():  # Exclude in_progress, manifests, and ocr_complete folders and log files
                continue
            self.process_folder(folder_name, folder_path)

    def process_folder(self, folder_name: str, folder_path: str) -> None:
        for tgz_filename in os.listdir(folder_path):
            if not tgz_filename.endswith('.tgz'):
                continue
            tgz_file_path = os.path.join(folder_path, tgz_filename)
            base_name = os.path.splitext(tgz_filename)[0]
            output_folder = os.path.join(self.output_dir, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            try:
                with tarfile.open(tgz_file_path, "r:gz") as outer_tar:
                    news_dict = self.process_tar(outer_tar)
            except tarfile.TarError as e:
                logging.error(f"Error extracting {tgz_filename}: {e}")
                continue
            output_file = os.path.join(output_folder, f"{base_name}.json")
            self.save_as_json(news_dict, output_file)

    def process_tar(self, outer_tar: tarfile.TarFile) -> Dict[str, Union[Dict[str, str], Dict[int, Dict[str, str]]]]:
        news_dict = {"newsletter_metadata": {}, "articles": {}}
        articles: Dict[int, Dict[str, str]] = {}
        id = 0
        for entry in outer_tar:
            try:
                if entry.name.endswith(".xml"):
                    file = outer_tar.extractfile(entry)
                    if file is not None:
                        content = file.read()
                        xml_content = content.decode('utf-8', 'ignore')
                        article = self.extract_article(xml_content, entry.name)
                        id += 1
                        news_dict["articles"][id] = article

                elif entry.name.endswith(".gz"):
                    gz_member = next(member for member in outer_tar.getmembers() if member.name.endswith('.gz'))
                    with outer_tar.extractfile(gz_member) as gz_file:
                        with gzip.open(gz_file, 'rt') as xml_file:
                            xml_string = xml_file.read()
                            newsletter_metadata = self.extract_meta(xml_string)
                            news_dict["newsletter_metadata"] = newsletter_metadata
                else:
                    continue
            except Exception as e:
                logging.error(f"Error processing file {entry.name}: {e}")
        return news_dict

    @staticmethod
    def save_as_json(data: Dict[str, Union[Dict[str, str], Dict[int, Dict[str, str]]]], output_file: str) -> None:
        try:
            with open(output_file, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        except Exception as e:
            logging.error(f"Error saving JSON to {output_file}: {e}")

    @staticmethod
    def extract_article(xml_content: str, file_name: str) -> Dict[str, str]:
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            logging.error(f"Failed to parse XML from file: {file_name}")
            return {}

        title_values = [element.text for element in root.iter() if element.tag.endswith('title')]
        if len(title_values) > 1:
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
        elif len(body_values) > 1:
            logging.warning("There are more than one paragraphs in the article.")
            body = ' '.join(body_values)
        else:
            body = body_values[0]

        return {"title": title, "body": body}

    @staticmethod
    def extract_meta(xml_string: str) -> Dict[str, Union[str, None]]:
        newsletter_metadata: Dict[str, Union[str, None]] = {}

        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError:
            logging.error("Failed to parse XML from file")
            return newsletter_metadata

        # Extracting metadata
        title_values = [element.text for element in root.iter() if element.tag.endswith('title')]
        if len(title_values)>1:
            logging.warning("More than one titles are extracted from metadata.")
        if not title_values:
            logging.warning("No title is extracted.")
            newsletter_metadata['title'] = None
        else:
            newsletter_metadata['title'] = title_values[0]

        language_values = [element.text for element in root.iter() if element.tag.endswith('language')]
        if len(language_values)>1:
            logging.warning("More than one language are extracted from metadata.")
        if not language_values:
            logging.warning("No language is extracted.")
            newsletter_metadata['language'] = None
        else:
            newsletter_metadata['language'] = language_values[0]


        issuenumber_values = [element.text for element in root.iter() if element.tag.endswith('issuenumber')]
        if len(issuenumber_values)>1:
            logging.warning("More than one issuenumbers are extracted from metadata.")
        if not issuenumber_values:
            logging.warning("No issuenumber is extracted.")
            newsletter_metadata['issuenumber'] = None
        else:
            newsletter_metadata['issuenumber'] = issuenumber_values[0]


        date_values = [element.text for element in root.iter() if element.tag.endswith('date')]
        if len(date_values)>1:
            logging.warning("More than one dates are extracted from metadata.")
        if not date_values:
            logging.warning("No date is extracted.")
            newsletter_metadata['date'] = None
        else:
            newsletter_metadata['date'] = date_values[0]

        identifier_values = [element.text for element in root.iter() if element.tag.endswith('identifier')]
        if len(identifier_values)>1:
            logging.warning("More than one identifiers are extracted from metadata.")
        if not identifier_values:
            logging.warning("No identifier is extracted.")
            newsletter_metadata['identifier'] = None
        else:
            newsletter_metadata['identifier'] = identifier_values[0]

        temporal_values = [element.text for element in root.iter() if element.tag.endswith('temporal')]
        if len(temporal_values)>1:
            logging.warning("More than one temporal are extracted from metadata.")
        if not temporal_values:
            logging.warning("No temporal is extracted.")
            newsletter_metadata['temporal'] = None
        else:
            newsletter_metadata['temporal'] = temporal_values[0]

        recordRights_values = [element.text for element in root.iter() if element.tag.endswith('recordRights')]
        if len(recordRights_values)>1:
            logging.warning("More than one recordRights are extracted from metadata.")
        if not recordRights_values:
            logging.warning("No recordRights is extracted.")
            newsletter_metadata['recordRights'] = None
        else:
            newsletter_metadata['recordRights'] = recordRights_values[0]

        publisher_values = [element.text for element in root.iter() if element.tag.endswith('publisher')]
        if len(publisher_values)>1:
            logging.warning("More than one publisher are extracted from metadata.")
        if not publisher_values:
            logging.warning("No publisher is extracted.")
            newsletter_metadata['publisher'] = None
        else:
            newsletter_metadata['publisher'] = publisher_values[0]

        spatial_values = [element.text for element in root.iter() if element.tag.endswith('spatial')]
        if len(spatial_values)>1:
            logging.warning("More than one spatial are extracted from metadata.")
        if not spatial_values:
            logging.warning("No spatial is extracted.")
            newsletter_metadata['spatial_1'] = None
            newsletter_metadata['spatial_2'] = None
        else:
            newsletter_metadata['spatial_1'] = spatial_values[0]
            newsletter_metadata['spatial_2'] = spatial_values[1]

        source_values = [element.text for element in root.iter() if element.tag.endswith('source')]
        if len(source_values)>1:
            logging.warning("More than one source are extracted from metadata.")
        if not source_values:
            logging.warning("No source is extracted.")
            newsletter_metadata['source'] = None
        else:
            newsletter_metadata['source'] = source_values[1]

        recordIdentifier_values = [element.text for element in root.iter() if element.tag.endswith('recordIdentifier')]
        if len(recordIdentifier_values)>1:
            logging.warning("More than one recordIdentifier are extracted from metadata.")
        if not recordIdentifier_values:
            logging.warning("No recordIdentifier is extracted.")
            newsletter_metadata['recordIdentifier'] = None
        else:
            newsletter_metadata['recordIdentifier'] = recordIdentifier_values[0]

        type_values = [element.text for element in root.iter() if element.tag.endswith('type')]
        if len(type_values)>1:
            logging.warning("More than one type are extracted from metadata.")
        if not type_values:
            logging.warning("No type is extracted.")
            newsletter_metadata['type'] = None
        else:
            newsletter_metadata['type'] = type_values[0]

        isPartOf_values = [element.text for element in root.iter() if element.tag.endswith('isPartOf')]
        if len(isPartOf_values)>1:
            logging.warning("More than one isPartOf are extracted from metadata.")
        if not isPartOf_values:
            logging.warning("No isPartOf is extracted.")
            newsletter_metadata['isPartOf_1'] = None
            newsletter_metadata['isPartOf_2'] = None
        else:
            newsletter_metadata['isPartOf_1'] = isPartOf_values[0]
            newsletter_metadata['isPartOf_2'] = isPartOf_values[1]

        return newsletter_metadata

# Configure logging
logging.basicConfig(filename='extractor.log', level=logging.DEBUG)

# Example usage
if __name__ == "__main__":
    extractor = XMLExtractor("../../data/news/gg", "../../data/news/gg-json")
    extractor.extract_xml_string()
