
| begin index | end index | format        | count | stride | use description                                                                                                                                    |
| ----------- | --------- | ------------- | ----- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0           | 6         | [[Resources]] | 1     | 6      | The absolute count of tokens belonging to the player                                                                                               |
| 6           | 11        | [[Resources]] | 1     | 5      | The count of resources provided to the player via cards                                                                                            |
| 11          | 12        | [[N-Hot]]         | 1     | 1      | The player's points                                                                                                                                |
| 12          | 13        | [[N-Hot]]         | 1     | 1      | Player ordering. If this is the first player to take a turn, 0. If this player took the last turn on the first round, will be equal to `playerN-1` |
| 13          | 46        | [[Card]]      | 3     | 11     | The player's three reserved cards                                                                                                                  |
<!-- TBLFM: $2=($1+($4*$5)) -->
<!-- TBLFM: @3$1..@>=@-1$2 -->