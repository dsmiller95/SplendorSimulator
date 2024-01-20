Represents an action to be taken by the current active player. All [[Ordered 1-Hot]] values are used to represent action preference. The turn executor will actually apply the action with the highest Turn type first, and will search all options within turns before attempting to the next turn type.

| begin index | end index | format                          | count | stride | name               | use description                                                                                                                                          |
| ----------- | --------- | ------------------------------- | ----- | ------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0           | 4         | [[Ordered 1-Hot]]               | 1     | 4      | Turn Type          | Which type of turn to take. Will attempt to take turns in order, executing the first turn which is valid to execute on the current game state.<br> |
| 4           | 19        | [[Ordered 1-Hot]]               | 1     | 15     | Card Pick          | A selection of a card on the board, for buying or purchasing. Uses same indexes as cards are placed in inside [[Game]]                                                                                            |
| 19          | 22        | [[Ordered 1-Hot]]               | 1     | 3      | Reserved Card Pick | A selection of a card in the players reserved cards.                                                                                                      |
| 22          | 27        | [[Ordered 1-Hot]]               | 1     | 5      | Noble Pick         | Which noble to choose from the nobles, when more than one available.                                                                                     |
| 27          | 37        | Ordered [[Combinatorial 1-Hot]] | 1     | 10     | Pick Three         | The combination of three tokens to select.                                                                                                               |
| 37          | 42        | [[Ordered 1-Hot]]               | 1     | 5      | Pick Two           | Which token to take two of                                                                                                                               |
| 42          | 126       | Ordered [[Combinatorial 1-Hot]] | 1     | 84     | Discard choice     | Which tokens to discard at the end of the players turn. Allows for picking between 0 and 3 tokens in any combination.                                    |
<!-- TBLFM: $2=($1+($4*$5)) -->
<!-- TBLFM: @3$1..@>=@-1$2 -->


## Turn types

- 0: Take Three unique tokens
- 1: Take Two of the same token
- 2: Purchase a card
- 3: Reserve a card
