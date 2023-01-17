import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import { GameState } from './models/gameState';
import { GetTokenFileName } from './models/resourceTokens';
import ResourceBank from './ResourceBank/ResourceBank';
import OpenCards from './OpenCards/OpenCards';
import Player from './Player/Player';
import GameDisplay from './GameDisplay/GameDisplay';

function App() {

  const [gameData, setData] = useState<GameState | null>(null)

  useEffect(() => {
    async function getGameData(): Promise<void> {

      var fetchResponse = await fetch('http://localhost:5000/game/json');
      var gameData: GameState = await fetchResponse.json();
      console.log(gameData);
      setData(gameData);
    }

    if(gameData == null){
      getGameData();
    }
  })


  return (
    <div className="App">
      <header className="App-header">
        {
          gameData == null ?
           <span>No Data</span> :
           <GameDisplay gameData={gameData}/>
        }
      </header>
    </div>
  );
}

export default App;
