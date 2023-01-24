export interface GameState {
  turn: number;
  nobles: Noble[];
  cards: Card[][];
  cardStacks: number[];
  bank: number[];
  players: PlayerState[];
  nextPlayer: number;
}

export interface Noble{
  id: number;
  costs: number[];
  points: number;
}

export interface Card{
  id: number;
  costs: number[];
  points: number;
  
  returns: number;
  tier: number;

}

export interface PlayerState{
  points: number;
  nobles: Noble[];
  tokens: number[];
  cards: Card[];
  card_resources: number[];
  reserved_cards: Card[];
}