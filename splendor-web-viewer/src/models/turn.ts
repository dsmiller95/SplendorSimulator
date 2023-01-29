export interface Turn {
  type: TurnType;
  discarded: number;
  resources_desired: number[];
  card_index: number;
  noble_preference: number;
  discard_preference: [number, number][];
}

export enum TurnType{
  TAKE_THREE_UNIQUE = "TAKE_THREE_UNIQUE",
  TAKE_TWO = "TAKE_TWO",
  BUY_CARD = "BUY_CARD",
  RESERVE_CARD = "RESERVE_CARD",
  NOOP = "NOOP"
}
