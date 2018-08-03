import QtQuick 2.0

Item {
    property bool scaleEnabled: true
    property bool routeEnabled: true
    property bool heliCentered: true
    signal routeEnableClicked
    signal scaleEnableClicked
    signal centerHeliClicked

    Image{
        x: 450
        y: 650
        id: trigger
        source: "icon-settings.png"
    }

    //the entire toolbar
    Rectangle{
        id:bar
        visible: false
        x: 420
        width: 80
        height: 300
        opacity: 0.6
        color:"lightgrey"

    }

    //button to enable altitude scale
    Rectangle{
        id:scaleEnable
        x: 460
        y:30
        width:30
        height:30
        color:"red"
        visible: false
    }
    Text{
        id:scaleEnableText
        x:scaleEnable.x -100
        y:scaleEnable.y -15
        visible: false
        text:"Enable altitude scale"
    }
    Text{
        id:scaleDisableText
        x:scaleEnable.x -100
        y:scaleEnable.y -15
        visible: false
        text:"Disable altitude scale"
    }



    //button to enable route and heli
    Rectangle{
        id:routeEnable
        x: 460
        y:90
        width:30
        height:30
        color:"yellow"
        visible: false
    }

    Text{
        id:showRoute
        x: routeEnable.x -60
        y: routeEnable.y -15
        visible: false
        text:"Show route"
    }
    Text{
        id:hideRoute
        x: routeEnable.x -60
        y: routeEnable.y -15
        visible: false
        text:"Hide route"
    }

    //button to center heli
    Rectangle{
        id:trackHeli
        x: 460
        y:150
        width:30
        height:30
        color:"green"
        visible: false
    }

    Text{
        id:centerHeli
	x: trackHeli.x -100
        y: trackHeli.y -15
        visible: false
        text:"Center Helicopter"
    }
    Text{
        id:decenterHeli
	x: trackHeli.x -100
        y: trackHeli.y -15
        visible: false
        text:"Decenter Helicopter"
    }


    //MouseArea to enable scale
    MouseArea{
        enabled: bar.visible
        hoverEnabled: true
        anchors.fill: scaleEnable
        onEntered:{
            if(scaleEnabled){
                scaleDisableText.visible = true;
            }else{
                scaleEnableText.visible = true;
            }
        }
        onExited:{
            scaleEnableText.visible = false;
            scaleDisableText.visible = false;
        }
        onClicked: {
            scaleEnableClicked()
            scaleEnabled = !scaleEnabled
            if(scaleEnabled){
                scaleDisableText.visible = true;
                scaleEnableText.visible = false;
            }else{
                scaleEnableText.visible = true;
                scaleDisableText.visible = false;
            }
        }
    }

    //MouseArea to show route
    MouseArea{
        enabled:bar.visible
        hoverEnabled: true
        anchors.fill: routeEnable
        onEntered:{
            if(routeEnabled){
                 hideRoute.visible = true;
            }else{
                showRoute.visible = true;
            }
        }
        onExited:{
            showRoute.visible = false;
            hideRoute.visible = false;
        }
        onClicked: {
            routeEnableClicked()
            routeEnabled = !routeEnabled
            if(routeEnabled){
                hideRoute.visible = true;
                showRoute.visible = false;
            }else{
                showRoute.visible = true;
                hideRoute.visible = false;
            }
        }
    }


    //MouseArea to center helicopter
    MouseArea{
        enabled:bar.visible
        hoverEnabled: true
        anchors.fill: trackHeli
        onEntered:{
            if(heliCentered){
                 decenterHeli.visible = true;
            }else{
                centerHeli.visible = true;
            }
        }
        onExited:{
            centerHeli.visible = false;
            decenterHeli.visible = false;
        }
        onClicked: {
            centerHeliClicked()
	    heliCentered = !heliCentered
            if(heliCentered){
                decenterHeli.visible = true;
                centerHeli.visible = false;
            }else{
                centerHeli.visible = true;
                decenterHeli.visible = false;
            }
        }
    }

    //mouseArea to trigger toolbar
    MouseArea{
        anchors.fill: trigger
        onClicked:{
            bar.visible = !bar.visible
            routeEnable.visible = !routeEnable.visible
            scaleEnable.visible = !scaleEnable.visible
            trackHeli.visible = !trackHeli.visible
        }
    }
}
