
from game_model.card import Card
from game_model.noble import Noble
from game_model.resource_types import ResourceType


import csv

class GameConfigData:
    def __init__(self, card_file_path: str) -> None:
        raw_rows = self.parse_csv(card_file_path)
        self.cards = self.ingest_cards(raw_rows)
        pass
    
    def parse_csv(self, file_path: str) -> list[list[str]]:
        data: list[list[str]] = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                data.append(row)
        return data

    def ingest_cards(self, data_table : list[list[str]]) -> list[Card]:
        result: list[Card] = []
        for row in data_table:
            if row[0] == "Card":
                card_level = row[1]
                card_costs =  row[2:6]
                reward_resource = ResourceType(row[7:12].index("1"))
                reward_points = row[13]

                result.append(Card(card_level, card_costs, reward_resource, reward_points))
        return result

    def ingest_nobles(self, data_table : list[list[str]]) -> list[Noble]:
        result: list[Noble] = []
        for row in data_table:
            if row[0] == "Noble":
                card_costs =  row[2:6]
                reward_points = row[13]

                result.append(Noble(card_costs, reward_points))
        return result