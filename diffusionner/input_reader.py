import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import List
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer

from diffusionner.entities import Dataset, EntityType, Entity, Document, DistributedIterableDataset

class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: AutoTokenizer, logger: Logger = None, repeat_gt_entities = None):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i+1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i+1] = entity_type

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger
        self._repeat_gt_entities = repeat_gt_entities

        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1

    @abstractmethod
    def read(self, datasets):
        pass

    def get_dataset(self, label):
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def _calc_context_size(self, datasets):
        sizes = [-1]

        for dataset in datasets:
            if isinstance(dataset, Dataset):
                for doc in dataset.documents:
                    sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: AutoTokenizer, logger: Logger = None, repeat_gt_entities = None):
        super().__init__(types_path, tokenizer, logger, repeat_gt_entities)

        
    def read(self, dataset_paths):
        for dataset_label, dataset_path in dataset_paths.items():
            if dataset_path.endswith(".jsonl"):
                dataset = DistributedIterableDataset(dataset_label, dataset_path, self._entity_types, tokenizer = self._tokenizer, repeat_gt_entities = self._repeat_gt_entities)
                self._datasets[dataset_label] = dataset
            else:
                dataset = Dataset(dataset_label, dataset_path, self._entity_types, tokenizer = self._tokenizer, repeat_gt_entities = self._repeat_gt_entities)
                self._parse_dataset(dataset_path, dataset, dataset_label)
                self._datasets[dataset_label] = dataset

        self._context_size = self._calc_context_size(self._datasets.values())

    def _parse_dataset(self, dataset_path, dataset, dataset_label):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset_label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset: Dataset) -> Document:
        jimages = None
        ltokens = None
        rtokens = None
        jrelations = None

        jtokens = doc['tokens']
        # jrelations = doc['relations']
        jentities = doc['entities']
        if "orig_id" not in doc:
            doc['orig_id'] = doc['org_id']
        orig_id = doc['orig_id']
        if "ltokens" in doc:
            ltokens = doc["ltokens"]

        if "rtokens" in doc:
            rtokens = doc["rtokens"]

        # parse tokens
        doc_tokens, doc_encoding, seg_encoding = self._parse_tokens(jtokens, ltokens, rtokens, dataset)

        if len(doc_encoding) > 512:
            self._log(f"Document {doc['orig_id']} len(doc_encoding) = {len(doc_encoding) } > 512, Ignored!")
            return None
        
        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)

        # create document
        document = dataset.create_document(doc_tokens, entities, doc_encoding, seg_encoding)

        return document


    def _parse_tokens(self, jtokens, ltokens, rtokens, dataset):
        doc_tokens = []
        special_tokens_map = self._tokenizer.special_tokens_map
        doc_encoding = [self._tokenizer.convert_tokens_to_ids(special_tokens_map['cls_token'])]
        seg_encoding = [1]

        if ltokens is not None and len(ltokens)>0:
            for token_phrase in ltokens:
                token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
                doc_encoding += token_encoding
                seg_encoding += [1] * len(token_encoding)
            doc_encoding += [self._tokenizer.convert_tokens_to_ids(special_tokens_map['sep_token'])]
            seg_encoding += [1]
        
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)

            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding) - 1 )
            token = dataset.create_token(i, span_start, span_end, token_phrase)
            doc_tokens.append(token)
            doc_encoding += token_encoding
            seg_encoding += [1] * len(token_encoding)
        
        if rtokens is not None and len(rtokens)>0:
            doc_encoding += [self._tokenizer.convert_tokens_to_ids(special_tokens_map['sep_token'])]
            seg_encoding += [1]
            for token_phrase in rtokens:
                token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
                # if len(doc_encoding) + len(token_encoding) > 512:
                #     break
                doc_encoding += token_encoding
                seg_encoding += [1] * len(token_encoding)

        return doc_tokens, doc_encoding, seg_encoding

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention  (exclusive)
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities