import React, { useEffect, useState } from 'react';
import './CardComponent.css';
import { GetTokenFileName } from '../models/resourceTokens';


interface CardProps {
  cardId?: number;
  imageOverride?: string;
}

function CardComponent(props: CardProps) {
  function getImageUrl(): string{
    if(props.imageOverride){
      return props.imageOverride;
    }
    if(props.cardId){
      return "game_assets/" + props.cardId.toString().padStart(3, "0") + ".png";
    }
    return "game_assets/Card-placeholder.png"
  }
  return (
    <div className="Card">
      <img src={getImageUrl()}/> 
    </div>
  );
}

export default CardComponent;
