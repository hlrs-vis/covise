import QtQuick 2.0

Item {
    property int thickLineNum
    property int thinLineNum
    property int minHeight
    property int maxHeight
    property string currentHeight
    property double pointerY
    property double textY
    property bool scaleVisible: true

    //the pointer that moves with altitude
    Rectangle {
        id: pointer
        width: 15
        height: 15
        x: 43
        y: pointerY
        z: 1
        color: "white"
        border.color: "black"
        border.width: 3
        radius: 10
        visible: scaleVisible
    }

    // the current height
    Text{
        visible: scaleVisible
        id: pText
        x: 80
        y: textY
        z: 2
        color: "black"
        font.pointSize: 13
        text: currentHeight
    }

    //thick number line
    Repeater {
        model: thickLineNum; // just define the number you want, can be a variable too

        delegate: Rectangle {
            visible: scaleVisible
            width: 35;
            height: 3;
            color: "black";
            x: 43
            y: 630 - index * 60;
            radius: 3;
        }
    }

    //the number on the number line
    Repeater {
        model: thickLineNum; // just define the number you want, can be a variable too

        delegate: Text {
            visible: scaleVisible
            text: minHeight + index * 100
            color: "black";
            x: 10
            y: 622 - index * 60;
        }
    }

    //thin number line
    Repeater {
        model: thinLineNum; // just define the number you want, can be a variable too

        delegate: Rectangle {
            visible: scaleVisible
            width: 20;
            height: 1.5;
            color: "black";
            x: 43
            y: 630 - index * 12;
            radius: 3;
        }
    }

    //the title
    Text{
        visible: scaleVisible
        x: 15
        y: 590 - (maxHeight - minHeight)*0.6
        z: 2
        color: "black"
        font.pointSize: 13
        text: "Altitude (ft)"
    }
}
