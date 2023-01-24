import React, { useEffect, useState } from 'react';
import { Card } from '../models/gameState';
import { GetTokenFileName } from '../models/resourceTokens';
import './PersistentPlayerDeck.css';

interface DeckProps{
  deck: Card[]
}

function GroupBy<ItemType, KeyType extends keyof ItemType>(items: ItemType[], filterKey: KeyType): Map<ItemType[KeyType], ItemType[]> {
  let seed = new Map<ItemType[KeyType], ItemType[]>();
  for(var item of items){
    let targetKey = item[filterKey];

    if(!seed.has(targetKey)){
      seed.set(targetKey, [])
    }

    seed.get(targetKey)?.push(item);
  }
  return seed;
}

function PlayerDeck(props: DeckProps) {

  let groupedCardsByReturns = GroupBy(props.deck, "returns");

  return (
    <div className="Player-deck">
        {
        Array.from(groupedCardsByReturns.entries()).map(([number, cardstack]) =>
          <div
          key={GetTokenFileName(number)}
          className="Card-type">
            <div>
              {
              cardstack.map((card) => 
              <div
                  key={card.id}
                  className="Card-img">
                  <img 
                  src={`game_assets/${card.id.toString().padStart(3, "0")}.png`} alt='game_assets/Card-placeholder.png'/>
              </div>)
              }
            </div>
          </div>)
        }
    </div>
  );
}
  
export default PlayerDeck;
