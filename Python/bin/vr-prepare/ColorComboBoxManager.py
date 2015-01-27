from ColorManager import ColorManager

class ColorComboBoxManager( ColorManager ):

    def __init__( self, parent=None ):
        """
           Handles combobox items from every VisItem panel
        """   
        ColorManager.__init__( self, parent )
        self.setupManagerUi()
        # necessary information referenced by the combobox item of the calling VisItem panel        
        self.__key2ColorIdx = {}
        self.__colorIdx2KeyIndex = {}
        
    def update( self, colorMapCombobox, currentVariable, currentColorTableKey=None ):
        cB = colorMapCombobox
        keyedColorList = ColorManager.getColorTables(self, currentVariable)
        
        colorMapCombobox.clear()
        currentIdx = None
        idx=0
        for keyColor in keyedColorList:
            self.__colorIdx2KeyIndex[(cB,idx)] = keyColor[0]
            self.__key2ColorIdx[(cB,keyColor[0])] = idx
            colorMapCombobox.addItem( keyColor[1] )
            if keyColor[1]==currentVariable:
                currentIdx = idx
            idx = idx + 1
        
        if currentColorTableKey and ((cB,currentColorTableKey) in self.__key2ColorIdx.keys()):
           currentIdx = self.__key2ColorIdx[(cB,currentColorTableKey)]
        if currentIdx:
            colorMapCombobox.setCurrentIndex(currentIdx)        

    def setSelectedColormapKey( self, colorMapCombobox, key ):
        try:
            colorMapCombobox.setCurrentIndex( self.__key2ColorIdx[(colorMapCombobox, key)] )
            return True
        except KeyError:
            return False
            
    def getSelectedColormapKey( self, colorMapCombobox ):
        return self.__colorIdx2KeyIndex[(colorMapCombobox, colorMapCombobox.currentIndex())]
