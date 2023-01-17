import React, { useEffect, useState } from 'react';
import './NobleComponent.css';
import { GetTokenFileName } from '../models/resourceTokens';


interface NobleProps {
  nobleId: number;
}

function NobleComponent(props: NobleProps) {
  var imageSource = "game_assets/" + props.nobleId.toString().padStart(3, "0") + ".png";
  return (
    <div className="Noble">
      <img src={imageSource}/> 
    </div>
  );
}

export default NobleComponent;
