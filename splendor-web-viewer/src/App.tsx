import React, { useEffect, useState } from 'react';
import logo from './logo.svg';
import './App.css';
import { GameState } from './models/gameState';
import { GetTokenFileName } from './models/resourceTokens';
import ResourceBank from './ResourceBank/ResourceBank';
import OpenCards from './OpenCards/OpenCards';
import Player from './Player/Player';
import GameDisplay from './GameDisplay/GameDisplay';
import PlayerDeck from './PersistentPlayerDeck/PersistentPlayerDeck';
import { Turn } from './models/turn';

interface AppState{
  game: GameState;
  gameIndex: number;
}

interface GameTurnResponse {
  game_state: GameState;
  turn_taken: Turn;
}

function App() {

  const [gameData, setGameData] = useState<GameState | null>(null)
  const [gameIndex, setGameIndex] = useState(0)

  useEffect(() => {
    async function getGameData(): Promise<void> {
      try{
        var fetchResponse = await fetch('http://localhost:5000/history/game/' + gameIndex.toString());
        if(!fetchResponse.ok){
          setGameData(null);
          return;
        }
  
        var gameData: GameTurnResponse = await fetchResponse.json();
        console.log(gameData);
        setGameData(gameData.game_state);
      }catch{
        setGameData(null);
      }
    }

    getGameData();
  }, [gameIndex])


  function nextGameState(){
    setGameIndex(gameIndex + 1);
  }
  function lastGameState(){
    setGameIndex(gameIndex - 1);
  }
  function setGameIndexFromText(indexStr: React.FormEvent<HTMLInputElement>){
    let index = parseInt(indexStr.currentTarget.value)
    if(Number.isNaN(index)){
      return;
    }
    setGameIndex(index);
  }

  return (
    <div className="App">
      <header className="App-header">
        <span className="Game-state-button" onClick={lastGameState}>Previous game state ({gameIndex - 1})</span>
        <input onChange={setGameIndexFromText}/>
        <span className="Game-state-button" onClick={nextGameState}>Next game state ({gameIndex + 1})</span>
      </header>
        {
          gameData == null ?
           <span>No Data</span> :
           <GameDisplay gameData={gameData}/>
        }
    </div>
  );
}

export default App;
