# Implement custom messages in depthai3

Existing depthai messages that inherit the Buffer class can be extended by:

```
import depthai as dai

class MyCustomMessage(dai.<MessageOfChoice>):
    def __init__(self):
        dai.<MessageOfChoice>.__init__(self)
        self.myField = 42

    def printMe(self):
        print("My field is", self.myField)
```

Note, this does NOT allow for passing to device and back. 