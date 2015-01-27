
from PyQt5 import QtCore, QtGui

import ObjectMgr

def SceneGraphItemPanelConnector(panel):
    #connections

    panel.nameWidget.returnPressed.connect(panel.emitNameChange)
    panel.vrpPushButtonColorRGB.clicked.connect(panel.emitColorRGB)
    panel.vrpPushButtonAmbientRGB_2.clicked.connect(panel.emitColorAmbient)
    panel.vrpPushButtonDiffuseRGB_2.clicked.connect(panel.emitColorDiffuse)
    panel.vrpPushButtonSpecularRGB_2.clicked.connect(panel.emitColorSpecular)
    panel.floatInRangeShininess_2.sigSliderReleased.connect(panel.emitDataChanged)
    panel.vrpCheckBoxTransparency.clicked.connect(panel.emitTransChecked)
    panel.floatInRangeTrans.sigSliderReleased.connect(panel.emitTransparencyChanged)
    panel.vrpRadioButtonNoColor.clicked.connect(panel.emitChangedRadioGroup)
    panel.vrpRadioButtonColorRGB.clicked.connect(panel.emitChangedRadioGroup)
    panel.vrpRadioButtonColorMaterial_2.clicked.connect(panel.emitChangedRadioGroup)
    """

    """
    

    panel.vrpCheckBoxIsMoveable.clicked.connect(panel.emitDataChanged)
    panel.vrpCheckBoxIsMoveSelected.clicked.connect(panel.emitDataChanged)
    
    ObjectMgr.ObjectMgr().sigGuiParamChanged.connect(panel.paramChanged)
    
    panel.shaderList.itemSelectionChanged.connect(panel.emitShaderChanged)    

    #validators:
    # allow only double values for transform lineEdits
    doubleValidator = QtGui.QDoubleValidator(panel)
    panel.floatX.setValidator(doubleValidator)
    panel.floatY.setValidator(doubleValidator)
    panel.floatZ.setValidator(doubleValidator)


def SceneGraphItemPanelBlockSignals( panel, doBlock ):
    panel.blockSignals(doBlock)
    # block all widgets with apply
    applyWidgets = [
        panel.vrpRadioButtonNoColor,
        panel.vrpRadioButtonColorRGB,
        panel.vrpPushButtonColorRGB,
        panel.vrpRadioButtonColorMaterial_2,
        panel.vrpPushButtonAmbientRGB_2,
        panel.vrpPushButtonDiffuseRGB_2,
        panel.vrpPushButtonSpecularRGB_2,
        panel.floatInRangeShininess_2,

        #panel.vrpRadioButtonColorVariable_2,
        #panel.vrpComboBoxVariable_2,
        #panel.vrpComboboxColorMap_2,
        #panel.vrpPushButtonCustomizeColorMap_2,

        panel.vrpCheckBoxTransparency,

        panel.vrpCheckBoxIsMoveable,
        panel.shaderList
    ]

    for widget in applyWidgets:
        widget.blockSignals(doBlock)
