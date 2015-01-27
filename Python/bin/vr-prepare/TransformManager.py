
from PyQt5 import QtCore, QtGui

import ObjectMgr
from Utils import getDoubleInLineEdit
from ChangeIndicatedLE import ChangeIndicatedLE

class TransformManager():

    def __init__(self, doApply, onChange=False):
        self.__hasScale = hasattr(self, 'floatScaleX')

        rotFloatInRangeCtrls = [ self.floatInRangeAxisX, self.floatInRangeAxisY, self.floatInRangeAxisZ  ]
        for w in rotFloatInRangeCtrls:
            w.setRange((-1.,1.))
            w.setValue(0.0)
        rotFloatInRangeCtrls.append(self.floatInRangeAngle)
        self.floatInRangeAngle.setRange([-180,180])
        self.floatInRangeAngle.setValue(0.0)

        changeIndicatedLEs = [self.floatX, self.floatY, self.floatZ]
        if self.__hasScale:
            changeIndicatedLEs.append(self.floatScaleX)
            changeIndicatedLEs.append(self.floatScaleY)
            changeIndicatedLEs.append(self.floatScaleZ)
            self.checkScaleUniform.toggled.connect(self.__uniformScaleClicked__)

        for c in changeIndicatedLEs:
            c.returnPressed.connect(doApply)
        for w in rotFloatInRangeCtrls:
            w.sigSliderReleased.connect(doApply)
            if onChange:
                w.sigValueChanged.connect(doApply)


    def __uniformScaleClicked__(self, checked):
        self.__uniformScaleUpdateUI__()
        if checked:
            # if uniform: emit signal since scaleY and scaleZ might have been changed in UpdateUI
            self.floatScaleX.returnPressed.emit()


    def __uniformScaleUpdateUI__(self):
        uniform = self.checkScaleUniform.isChecked()
        self.floatScaleY.setEnabled(not uniform)
        self.floatScaleZ.setEnabled(not uniform)
        if uniform:
            self.floatScaleY.setText(self.floatScaleX.text())
            self.floatScaleZ.setText(self.floatScaleX.text())


    def TransformManagerBlockSignals( self, doBlock ):
        # block all widgets with apply
        applyWidgets = [
            self.floatInRangeAxisX,
            self.floatInRangeAxisY,
            self.floatInRangeAxisZ,
            self.floatInRangeAngle,
            self.floatX,
            self.floatY,
            self.floatZ
        ]
        if self.__hasScale:
            applyWidgets.append(self.floatScaleX)
            applyWidgets.append(self.floatScaleY)
            applyWidgets.append(self.floatScaleZ)
        for widget in applyWidgets:
            widget.blockSignals(doBlock)


    def TransformManagerSetParams(self, params):
        if hasattr(params, 'rotAngle'):
            self.floatInRangeAxisX.setValue(params.rotX)
            self.floatInRangeAxisY.setValue(params.rotY)
            self.floatInRangeAxisZ.setValue(params.rotZ)
            if (params.rotAngle <= 180.0) and (params.rotAngle >= -180.0):
                self.floatInRangeAngle.setValue(params.rotAngle)
            elif params.rotAngle > 180.0:
                self.floatInRangeAngle.setValue(180.0 - params.rotAngle)
            else:
                self.floatInRangeAngle.setValue(180.0 + params.rotAngle)
            self.floatX.setText(str(params.transX))
            self.floatY.setText(str(params.transY))
            self.floatZ.setText(str(params.transZ))
            if self.__hasScale:
                self.floatScaleX.setText(str(params.scaleX))
                self.floatScaleY.setText(str(params.scaleY))
                self.floatScaleZ.setText(str(params.scaleZ))
                # set the uniform CheckBox
                self.checkScaleUniform.setChecked((params.scaleX == params.scaleY) and (params.scaleX == params.scaleZ))
                self.__uniformScaleUpdateUI__()


    def TransformManagerGetParams(self, params):
        params.rotX = self.floatInRangeAxisX.getValue()
        params.rotY = self.floatInRangeAxisY.getValue()
        params.rotZ = self.floatInRangeAxisZ.getValue()
        params.rotAngle = self.floatInRangeAngle.getValue()
        params.transX = getDoubleInLineEdit(self.floatX)
        params.transY = getDoubleInLineEdit(self.floatY)
        params.transZ = getDoubleInLineEdit(self.floatZ)
        if self.__hasScale:
            params.scaleX = getDoubleInLineEdit(self.floatScaleX)
            if self.checkScaleUniform.isChecked():
                params.scaleY = params.scaleX
                params.scaleZ = params.scaleX
            else:
                params.scaleY = getDoubleInLineEdit(self.floatScaleY)
                params.scaleZ = getDoubleInLineEdit(self.floatScaleZ)
        else:
            params.scaleX = 1.0
            params.scaleY = 1.0
            params.scaleZ = 1.0


    #def getCurrentTransformation(panel):
        #transformation=[]
        #transformation.append(panel.floatInRangeAxisX.getValue())
        #transformation.append(panel.floatInRangeAxisY.getValue())
        #transformation.append(panel.floatInRangeAxisZ.getValue())
        #transformation.append(panel.floatInRangeAngle.getValue())
        #transformation.append(getDoubleInLineEdit(panel.floatX))
        #transformation.append(getDoubleInLineEdit(panel.floatY))
        #transformation.append(getDoubleInLineEdit(panel.floatZ))
        #return transformation
