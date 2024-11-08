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
        y: 630
        id: trigger
        source: "icon-settings.png"
    }

    //the entire toolbar
    Rectangle{
        id:bar
        visible: false
        x: 440
        width: 60
        height: 300
        opacity: 0.6
        color:"lightgrey"

    }

    //button to enable altitude scale
    Image{
        id:scaleEnable
        x: 460
        y:25
        visible: false
        source: "height.png"
    }
    Text{
        id:scaleEnableText
        x:scaleEnable.x -50
        y:scaleEnable.y -20
        visible: false
        text:"Enable scale"
    }
    Text{
        id:scaleDisableText
        x:scaleEnable.x -50
        y:scaleEnable.y -20
        visible: false
        text:"Disable scale"
    }



    //button to enable route and heli
    Image{
        id:routeEnable
        x: 460
        y:85
        visible: false
        source: "route.png"
    }

    Text{
        id:showRoute
        x: routeEnable.x -30
        y: routeEnable.y -20
        visible: false
        text:"Show route"
    }
    Text{
        id:hideRoute
        x: routeEnable.x -30
        y: routeEnable.y -20
        visible: false
        text:"Hide route"
    }

    //button to center heli
    Image{
        id:trackHeli
        x: 460
        y:145
        visible: false
        source: "center.png"
    }

    Text{
        id:centerHeli
	x: trackHeli.x -50
        y: trackHeli.y -20
        visible: false
        text:"Center Heli"
    }
    Text{
        id:decenterHeli
	x: trackHeli.x -50
        y: trackHeli.y -20
        visible: false
        text:"Decenter Heli"
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
