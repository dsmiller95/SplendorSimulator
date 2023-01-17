import './Player.css';
import { Card, PlayerState } from '../models/gameState';
import CardComponent from '../CardComponent/CardComponent';
import NobleComponent from '../NobleComponent/NobleComponent';
import ResourceBank from '../ResourceBank/ResourceBank';

interface PlayerProps{
  player: PlayerState;
  playerIndex: number;
  isActive: boolean;
}

function Player(props: PlayerProps) {
  return (
    <div
      className={'Player-area ' + (props.isActive ? 'Player-active' : '')}>
        <h1>{`Player ${props.playerIndex}. Points: ${props.player.points}`}</h1>
        <div
          className='Noble-list'
        >
          {
            props.player.nobles.map((noble, i) => <NobleComponent key={i} nobleId={noble.id}/>)
          }
        </div>
        <div
          className='Token-pool'
        >
          <ResourceBank resources={props.player.tokens}/>
        </div>
        <div
          className='Persistent-resources'
        >
          <ResourceBank resources={props.player.card_resources}/>
        </div>
        <div
          className='Reserved-cards'
        >
          {
            props.player.reserved_cards.map((card, i) => <CardComponent key={i} cardId={card?.id}/>)
          }
        </div>
    </div>
  );
}

export default Player;
