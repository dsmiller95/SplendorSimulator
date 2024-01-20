Outlines the layout of the Game State output vector

| begin index | end index | format        | count | stride | use description                                                                    |
| ----------- |:--------- | ------------- | ----- | ------ | ---------------------------------------------------------------------------------- |
| 0           | 184       | [[Player]]    | 4     | 46     | Public player data. Index 0 is the current player, and Index 1 is the next player. |
| 184         | 214       | [[Noble]]     | 5     | 6      | All nobles currently available                                                     |
| 214         | 220       | [[Resources]] | 1     | 6      | The resources currently available in the bank                                      |
| 220         | 385       | [[Card Row]]  | 3     | 55     | All three tiers of cards. Index 0 is the first tier (lowest cost cards) and Index 2 is the top tier.                                                           |
<!-- TBLFM: $2=($1+($4*$5)) -->
<!-- TBLFM: @3$1..@>=@-1$2 -->