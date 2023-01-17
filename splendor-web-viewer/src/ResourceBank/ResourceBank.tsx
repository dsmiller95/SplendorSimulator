import React, { useEffect, useState } from 'react';
import './ResourceBank.css';
import { GetTokenFileName } from '../models/resourceTokens';

interface BankProps{
    resources: number[]
}

function ResourceBank(props: BankProps) {
  return (
    <div className="Resource-bank">
        {
        props.resources.map((n, index) => 
        <div
          key={index}
          className="Resource-type">
            <div>
              {Array.from({length: n}, (_, next_index) => 
              <div
                  key={index * 100 + next_index}
                  className="Resource-token">
                  <img 
                  src={`game_assets/${GetTokenFileName(index)}.png`}/>
              </div>)}
            </div>
        </div>)
        }
    </div>
  );
}

export default ResourceBank;
