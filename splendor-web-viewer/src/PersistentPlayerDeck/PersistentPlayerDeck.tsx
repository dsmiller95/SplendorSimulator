import React, { useEffect, useState } from 'react';
import { Card } from '../models/gameState';
import CardComponent from '../CardComponent/CardComponent';

interface DeckProps{
    deck: Card[]
}

// "Most efficient method to groupby on an array of objects"; https://stackoverflow.com/a/34890276 by https://stackoverflow.com/users/577199/ceasar
// groupBy(['one', 'two', 'three'], 'length') => {"3": ["one", "two"], "5": ["three"]}
var GroupBy = function(array: Card[], key: Number) {
  return array.reduce(function(new_arr: {}, elem: Card[]) {
      (new_arr[elem[key]] = new_arr[elem[key]] || []).push(elem);
      return new_arr;},{});
};

function PlayerDeck(props: DeckProps) {
  return (
    <div className="Player-deck">
        {
        props.deck.map(GroupBy(DeckProps.deck, DeckProps.deck.returns.indexOf(1))) => 
        <div
          key={props.} //find the index of the returned token type number eg. [0,0,1,0,0,0] -> 2
          className="Card-type">
            <div>
              {Array.from({length: card}, (_, next_index) => 
              <div
                  key={index * 100 + next_index}
                  className="Card-img">
                  <img 
                  src={`game_assets/${"put card filename here"}.png`}/>
              </div>)}
            </div>
        </div>)
        }
    </div>
  );
}

export default PlayerDeck;
