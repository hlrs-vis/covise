from coGRMsg import *

def simpleTest():
    msg = coGRObjVisMsg("COLLECT_1_001_0", 1)
    c = msg.c_str()
    c = "GRMSG\n0\nTTT\n1"
    newmsg = coGRMsg(c)
    if newmsg.getType()==coGRMsg.VISIBLE:
        newvismsg = coGRObjVisMsg(c)
    
    assert newvismsg.isValid(), 'not valid'
    assert newvismsg.isVisible(), 'test failed'

simpleTest()       

