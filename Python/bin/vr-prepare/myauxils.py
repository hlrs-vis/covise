
"""Various helpful python entities.


Copyright (c) 2006 Visenso GmbH

OneToOne -- class realizing a bijection

"""

def cLikeCounter():
    count = 0
    while True:
        yield count
        count += 1


class OneToOne(object):

    """Maintain a bijection.

    The bijection goes from a domain onto a codomain.

    TODO(improve):
    . Find a better interface and realization of this
    class.  Hint: there are suggestions on the web for
    realizing bijections within python.

    """

    def __init__(self):
        self.__mapping = {}
        self.__inverse = {}

    def __str__(self):
        return self.__mapping


    def has_preimage(self, testee):
        """Expected to be faster than searching testee in domain."""
        return testee in self.__mapping

    def has_image(self, testee):
        """Expected to be faster than searching testee in codomain."""
        return testee in self.__inverse

    def preimage(self, element):
        #assert element in self.__inverse.keys() # very time consuming for large dicts
        return self.__inverse[element]

    def image(self, element):
        #assert element in self.__mapping.keys() # very time consuming for large dicts
        return self.__mapping[element]

    def domain(self):
        return self.__mapping.keys()

    def codomain(self):
        return self.__inverse.keys()


    def insert(self, pair):

        """Extend the bijection.

        Precondition: The object continues to be a one-to-one-mapping.

        """

        preimage = pair[0]
        image = pair[1]

        #assert not preimage in self.__mapping, \
               #'preimage %s already exists.'
        #assert not image in self.__inverse, \
               #'image %s already exists.'
               
        # dont be so restrictive with already existing pairs.
        # just dont add again instead of assertion error!
        if preimage in self.__mapping and image in self.__inverse:
            if self.__mapping[preimage] == image and self.__inverse[image] == preimage:
                return
        elif preimage in self.__mapping or image in self.__inverse:
            assert False, 'Similar bijection entry with equal (pre)image already exists!'
        
        self.__mapping[preimage] = image
        self.__inverse[image] = preimage

    def remove(self, preimage):

        """Remove the preimage-image pair from the bijection.

        Precondition: preimage lies in self.domain().

        """

        assert preimage in self.domain()

        image = self.image(preimage)
        del self.__mapping[preimage]
        del self.__inverse[image]


class KeyHandler(object):

    """Register arbitrary objects and get a unique key as reference."""

    def __init__(self):
        self.__partKey = -1
        self.__key2Instance = {}

    def hasKey( self, key):
        return key in self.__key2Instance

    def getObject(self, key):
        if self.hasKey(key):
            return self.__key2Instance[key]
        return None

    def registerObject( self, obj):
        self.__partKey += 1
        assert (not self.hasKey(self.__partKey))
        self.__key2Instance[self.__partKey] = obj
        return self.__partKey

    def registerKeydObject( self, key, obj):
        if self.hasKey(key):
            print("WARNING: key of registered object already exists")
        self.__key2Instance[key] = obj
        self.__partKey = max( self.__key2Instance.keys() )

    def unregisterObject(self, key):
        if self.hasKey(key):
            del self.__key2Instance[key]

    # function only used in vr-prepare-v2
    def deletePartOfKey(self, key):
        self.unregisterObject(key)

    def getOffset(self):
        # return next free partNumber to add a saved project
        return self.__partKey+1

    def getAllElements(self):
        return self.__key2Instance
    
    def delete(self):
        self.__partKey = -1
        self.__key2Instance = {}

class NamedCheckable(object):
    def __init__(self, name='', isChecked=False):
        self.name = name
        self.isChecked = isChecked
    def __eq__(self, other):
        return (
            self.name == other.name and
            self.isChecked == other.isChecked)
    def __ne__(self, other):
        return not self == other

# eof
