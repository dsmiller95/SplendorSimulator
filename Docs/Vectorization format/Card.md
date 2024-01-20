
| begin index | end index | format        | count | stride | use description                 |
| ----------- | --------- | ------------- | ----- | ------ | ------------------------------- |
| 0           | 5         | [[Resources]] | 1     | 5      | Cost of the card                |
| 5           | 10        | [[Resources]] | 1     | 5      | Resources provided by the card  |
| 10          | 11        | [[N-Hot]]         | 1     | 1      | The points rewarded by the card |
<!-- TBLFM: $2=($1+($4*$5)) -->
<!-- TBLFM: @3$1..@>=@-1$2 -->
