var tab = new coTUITab("ScriptExample", tui.mainpanel.id);
tab.setPos(0, 0);

var button = new coTUIButton("Click Me!", tab.id);
var edit = new coTUIEditTextField("0", tab.id);

button.setPos(0, 0);
edit.setPos(0, 1);

button.tabletReleaseEvent.connect(function() { edit.text = parseInt(edit.text) + 1; });

