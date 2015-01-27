"""File: CoviseMsgLoop.py

TODO: Realize style-guide conformance.  Quote:
'Modules should have short, lowercase names, without
underscores.'

Description:

Interface to access the covise msg-loop. In order to
add an user-defined action derive a class from
CoviseMsgLoopAction and whatever should be done in
the run method. Be aware that the run method is
executed in the main-loop of the receiver
thread. Therefore the run method should return
immediatly otherwise the whole scripting interface is
blocked. The general MsgLoopAction is created that
way that ALL actions registered for one type are
executed.

Status: under dev UNRELEASED

(C) 2003 VirCinity GmbH, Stuttgart   info@vircinity.com
(C) 2004-2006 Visenso GmbH, Stuttgart   info@visenso.de

Initial version (date, author): 15.07.2004 [RM]

"""

from printing import InfoPrintCapable


class CoviseMsgLoopAction:

    """Base of actions triggered after receiving a covise-msg.

    You can register a CoviseMsgLoopAction in the
    CoviseMsgLoop().  The run-method of the action is
    then called whenever a covise-message of type type
    occurs.

    Central is the type of the covise-message.  Find
    type-numbers for covise-messages in file
    $COVISEDIR/src/kernel/covise/covise_msg.h in the
    definition of enum covise_msg_type.

    """
    runType_=None

    def __init__(self, name, type, desc=''):

        """
        name -- name of the action
        type -- covise-message type number
        desc -- description of the action

        """

        self.__desc = desc
        if not isinstance(type,tuple):
            self.__type = (type,)
        else:
            self.__type = type
        self.__name = name

    def run(self, param=None):
        print("CoviseMsgLoopAction.run(): redefine for specific functionality")

    def type(self): return self.__type
    def name(self): return self.__name
    def desc(self): return self.__desc

    def __str__(self):
        ret='CoviseMsgLoopAction: type: '
        for t in self.__type:
            ret += str(t)+' '
        ret += (' name: %s, description: %s.'% (self.__name, self.__desc)) 
        return ret
                


class _CoviseMsgLoop:

    """Heart of the Action singleton.

    The talkativeness of this class can be switched on.

    """

    instance_ = None

    def __init__(self):
        self.__actionDict = {}
        infoer = InfoPrintCapable()
        # infoer.doprint(= True
        infoer.module = self.__module__
        infoer.class_ = self.__class__.__name__
        self.__infoer = infoer

    def register(self, anAction):

        """Register action anAction in the main-loop.

        Actions are now multi-actions by default. It is possible to
        use a tuple for the type field. The action will hbe used for
        each msg. type in the tuple.  If an int is given the action is
        properly handled anyway. 

        Ignore register trys of aready registered
        actions.  Comparison of two actions relies on
        their type and name.

        """

        assert anAction

        for typ in anAction.type():
            if typ in self.__actionDict:
                actions = self.__actionDict[typ]
                for action in actions:
                    if action.name() == anAction.name():
                        return                
                actions.append(anAction)
                self.__actionDict[typ] = actions
            else: # No action for this anAction.type() yet.
                self.__actionDict[typ] = [anAction]

    def unregister(self, anAction):

        """Remove an anAction from this instance.

        Return immediatly when anAction is not registered.

        """
        for typ in anAction.type():
            if not typ in self.__actionDict:
                return
            actions = self.__actionDict[typ]
            if not anAction in actions:
                return
            actions.remove(anAction)

    def run(self, type, param=None):

        """Run all registered actions of type type with param as argument."""

        infoer = self.__infoer
        infoer.function = self.run.__name__
        handled = False
        if type in self.__actionDict:
            actions = self.__actionDict[type]
            for action in actions:
                try:
                    action.runType_=type
                    action.run(param)
                    handled = True
                    infoer.write("Msg handled by action '%s'" % str(action) )
                except IndexError:
                    infoer.startString = '(receiver index exception caught)'
                    infoer.write("(action '%s', params '%s')" % (
                        str(action), str(param)))
                except KeyError:
                    infoer.startString = '(receiver key exception caught)'
                    infoer.write("(action '%s', params '%s')" % (
                        str(action), str(param)))
                except MemoryError:
                    infoer.startString = '(receiver memory exception caught)'
                    infoer.write("(action '%s', params '%s')" % (
                        str(action), str(param)))
                except NameError:
                    infoer.startString = '(receiver name exception caught)'
                    infoer.write("(action '%s', params '%s')" % (
                        str(action), str(param)))
        else:
            infoer.startString = '(info)'
            infoer.write("No action found to run for covise-message-type <"
                         + str(type) + ">.")
        return handled

def CoviseMsgLoop():

    """Return the _CoviseMsgLoop-singleton.

    This function shall be the only accessor to class
    _CoviseMsgLoop.  This convention realizes
    _CoviseMsgLoop as singleton.

    """

    if _CoviseMsgLoop.instance_ == None:
        _CoviseMsgLoop.instance_ = _CoviseMsgLoop()
    return _CoviseMsgLoop.instance_

# eof

