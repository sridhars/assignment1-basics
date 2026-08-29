unicode 1:
(a) '\x00'

(b) its a nested string: "'\x00'"

(c) when printed, its not rendered. but in the bytes of the string, it is stored as '\x00'

unicode 2:
(a) utf 8 uses fewer bytes for the same string 

(b) "helloこんに"

(c) [255, 255] or [0xFF, 0xFF]

