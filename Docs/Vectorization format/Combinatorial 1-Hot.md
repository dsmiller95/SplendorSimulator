Similar to [[1-Hot]], but specifically used to establish an ordering among multiple options rather than picking exactly one. And the chosen index in the 1-Hot array is used to reconstruct a possible selection among a combinatorial list.

For example, suppose we want to represent all possible ways to pick 3 items out of a set of 5 without picking any duplicates, and represent this as 1-Hot. There are exactly 10 unique ways to do this. So we establish an ordering over all these options, and use their indexes in the 1-Hot array.

When [[Ordered 1-Hot]], will also establish ordering over all possible option.