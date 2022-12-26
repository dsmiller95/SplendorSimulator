from __future__ import annotations
from game_model.card import Card
from game_model.noble import Noble
from game_model.resource_types import ResourceType
import csv

from utilities.subsamples import parse_all_int, parse_int

class GameConfigData:
    def __init__(self, cards: list[Card], nobles: list[Noble]) -> None:
        self.cards = cards
        self.nobles = nobles

        self.open_cards_per_tier = 4
        self.max_reserved_cards = 3
        self.max_resource_tokens = 10
        self.tiers = 3
        pass
    
    def total_available_cards(self) -> int:
        return self.tiers * self.open_cards_per_tier

    @staticmethod
    def read_file(card_file_path: str) -> GameConfigData:
        raw_rows = GameConfigData.parse_csv(card_file_path)
        return GameConfigData(
            cards=GameConfigData.ingest_cards(raw_rows),
            nobles=GameConfigData.ingest_nobles(raw_rows)
        )
    
    @staticmethod
    def parse_csv(file_path: str) -> list[list[str]]:
        data: list[list[str]] = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                data.append(row)
        return data

    @staticmethod
    def ingest_cards(data_table : list[list[str]]) -> list[Card]:
        result: list[Card] = []
        for row in data_table:
            if row[1] == "Regular":
                reward_index = row[8:13].index("1")

                result.append(Card(
                    card_level= int(row[2]) - 1,
                    resource_cost=parse_all_int(row[3:8], 0),
                    reward_resource=ResourceType(reward_index),
                    reward_points=parse_int(row[13], 0),
                    card_id=int(row[0])))
        return result

    @staticmethod
    def ingest_nobles(data_table : list[list[str]]) -> list[Noble]:
        result: list[Noble] = []
        for row in data_table:
            if row[1] == "Noble":
                result.append(Noble(
                    resource_cost=parse_all_int(row[3:8], 0),
                    reward_points=parse_int(row[13], 0)))
        return result