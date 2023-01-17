import React, { useEffect, useState } from 'react';
import './NobleComponent.css';
import { GetTokenFileName } from '../models/resourceTokens';


interface NobleProps {
  nobleId: number;
  imageOverride?: string;
}

function NobleComponent(props: NobleProps) {
  function getImageUrl(): string{
    if(props.imageOverride){
      return props.imageOverride;
    }
    if(props.nobleId){
      return "game_assets/" + props.nobleId.toString().padStart(3, "0") + ".png";
    }
    return "game_assets/Noble-placeholder.png"
  }
  return (
    <div className="Noble">
      <img src={getImageUrl()}/> 
    </div>
  );
}

export default NobleComponent;
