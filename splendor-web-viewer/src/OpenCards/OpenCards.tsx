import './OpenCards.css';
import { Card } from '../models/gameState';
import CardComponent from '../CardComponent/CardComponent';

interface OpenCardsProps{
  cardsByTier: Card[][];
  remainingCardsPerTier: number[];
}

function OpenCards(props: OpenCardsProps) {
  var totalTiers = [0, 1, 2];
  return (
    <div
      className='All-cards'>
      {
        totalTiers.map(tier => 
          <div 
          key={tier}
          className="Card-tier">
            <CardComponent imageOverride={`game_assets/Tier${tier + 1}.png`}/>
            {
              props.cardsByTier[tier].map((card, i) => <CardComponent key={i} cardId={card.id}/>)
            }
          </div>)
      }
    </div>
  );
}

export default OpenCards;
