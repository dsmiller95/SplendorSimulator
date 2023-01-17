import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import { GameState } from './models/gameState';
import { GetTokenFileName } from './models/resourceTokens';
import ResourceBank from './ResourceBank/ResourceBank';

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
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        {
          gameData == null ?
           <span>No Data</span> :
           <div>
            <ResourceBank resources={gameData.bank}/>
            <pre>{JSON.stringify(gameData)}</pre>
           </div>
        }
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
