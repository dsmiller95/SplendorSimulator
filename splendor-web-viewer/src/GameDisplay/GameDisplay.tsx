import './GameDisplay.css';
import { Card, GameState, PlayerState } from '../models/gameState';
import ResourceBank from '../ResourceBank/ResourceBank';
import OpenCards from '../OpenCards/OpenCards';
import Player from '../Player/Player';
import NobleComponent from '../NobleComponent/NobleComponent';

interface GameProps {
  gameData: GameState;
}

function GameDisplay(props: GameProps) {
  var gameData = props.gameData;
  return (
    <div>
      <div
        className='Nobles'
      >
      {
        gameData.nobles.map((x, i) => <NobleComponent key={i} nobleId={x.id}/> )
      }
      </div>
      <OpenCards cardsByTier={gameData.cards} remainingCardsPerTier={gameData.cardStacks} />
      <ResourceBank resources={gameData.bank} />
      <div
        className='Players'
      >
        {
          gameData.players.map((x, i) => <Player key={i} player={x} playerIndex={i} />)
        }
      </div>
    </div>
  );
}

export default GameDisplay;
