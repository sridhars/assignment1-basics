## BPE

### unicode1
(a) '\x00'
(b) __repr__() is escaped and include the quotes. it is literally what is rendered to the screen
(c) chr(0) -> '\x00'
    print(chr(0)) -> nothing
    "this is a test" + chr(0) + "string" -> 'this is a test\x00string'
    print("this is a test" + chr(0) + "string") -> 'this is a teststring'

TODO - understand above: __repr__ vs print

### unicode2
(a) UTF-8 is more compact. . It uses fewer bytes for commonly used characters. Since attention is quadratic in terms of sequence length, this matters.
(b) For a UTF-8 encoded string, one character may be represented by one to four bytes. This decode assumes characters can be decoded atomically.
(c) '\xff\xff' - TODO verify



