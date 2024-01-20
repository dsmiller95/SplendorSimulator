An array of values. Represents a single integer value, typically as a method of selecting from a list of options. The number of options is equal to the size of the array. The index of the array element with the greatest value is the option "chosen".

For example, a 1-Hot of size 7 the chosen Option could be between 0 and 6 inclusive. In this example:  `[1, 3, 4, 0, -20, 10, 3]` the chosen Option is 5.

In some cases, the list of options is used as a way to establish an ordering over all options. For example with a 1-Hot vector if size 4: `[0.3, 1.4, 0.1, 0.5]`  we can establish an ordering of options: 1, 3, 0, 2