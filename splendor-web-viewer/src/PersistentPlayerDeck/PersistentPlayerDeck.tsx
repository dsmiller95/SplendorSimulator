import React, { useEffect, useState } from 'react';
import { Card } from '../models/gameState';
import { GetTokenFileName } from '../models/resourceTokens';
import './PersistentPlayerDeck.css';

interface DeckProps{
    deck: Card[]
}

// "Most efficient method to groupby on an array of objects"; https://stackoverflow.com/a/34890276 by https://stackoverflow.com/users/577199/ceasar
// groupBy(['one', 'two', 'three'], 'length') => {"3": ["one", "two"], "5": ["three"]}
// var GroupBy = function(array: Card[], key: Number) {
//   return array.reduce(function(new_arr: {}, elem: Card[]) {
//       (new_arr[elem[key]] = new_arr[elem[key]] || []).push(elem);
//       return new_arr;},{});
// };
var CardsByResource = function(deck: Card[]){
  var cardsByResource: Card[][] = [[], [], [], [], []];
  for(var card of deck){
    var cardReturnType = card.returns;
    if(cardReturnType < 0){
      console.error("card with no valid return type found. returns: " + card.returns);
      continue;
    }
    cardsByResource[cardReturnType].push(card);
  }
  return cardsByResource
}
function PlayerDeck(props: DeckProps) {
  return (
    <div className="Player-deck">
        {
        CardsByResource(props.deck).map((cardstack,index) =>
          <div
          key={GetTokenFileName(index)}
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
