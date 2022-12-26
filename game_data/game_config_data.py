from __future__ import annotations
from game_model.card import Card
from game_model.noble import Noble
from game_model.resource_types import ResourceType
import csv

class GameConfigData:
    def __init__(self, cards: list[Card], nobles: list[Noble]) -> None:
        self.cards = cards
        self.nobles = nobles

        self.open_cards_per_tier = 4
        pass
    

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
            if row[0] == "Card":
                card_level = row[1]
                card_costs =  row[2:6]
                reward_resource = ResourceType(row[7:12].index("1"))
                reward_points = row[13]

                result.append(Card(card_level, card_costs, reward_resource, reward_points))
        return result

    def ingest_nobles(data_table : list[list[str]]) -> list[Noble]:
        result: list[Noble] = []
        for row in data_table:
            if row[0] == "Noble":
                card_costs =  row[2:6]
                reward_points = row[13]

                result.append(Noble(card_costs, reward_points))
        return result