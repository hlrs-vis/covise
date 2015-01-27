var sliderWidth = 200;
var sliders = new Object;
var uuid = "";
var controlPanelMode = true;
var running = true;

var categories = new Array;
var modulesInCategory = new Object;

var mappedModuleHeaders = new Object;

var colormapChoices = new Object;

function init()
{
   addEventListener();
   listHosts();
   getRunningModules();
}

function stop()
{
   running = false;
   removeEventListener();
}

function removeEventListener()
{
   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:removeEventListener><uuid>' + uuid + '</uuid></covise:removeEventListener></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, false);
}


function addEventListener()
{
   if (uuid === "")
   {
      uuid = "_requested_";
      var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:addEventListener></covise:addEventListener></SOAP-ENV:Body></SOAP-ENV:Envelope>';
      sendSOAPRequest(request, true, addEventListenerEnd);
   }

}

function addEventListenerEnd(httpRequest)
{

   if (httpRequest.readyState == 4) {
      if (httpRequest.status == 200) {
         uuid = httpRequest.responseXML.documentElement.firstElementChild.firstElementChild.firstElementChild.firstChild.wholeText;
         getEvent();
      } else {
         uuid = "";
         //if (running === true)
         //   alert('There was a problem with the request.');
      }
   }
}

function listHosts()
{
   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"> <SOAP-ENV:Body><covise:listHosts></covise:listHosts></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, true, listHostsEnd);
}

function listHostsEnd(httpRequest)
{

   if (httpRequest.readyState == 4) {
      if (httpRequest.status == 200) {
         var hosts = httpRequest.responseXML.documentElement.getElementsByTagName('hosts');
         for (var host = 0; host < hosts.length; ++host)
         {
            var element = document.createElement('option');
            var content = document.createTextNode(hosts[host].textContent);
            element.appendChild(content);
            document.getElementById('hostselect').appendChild(element);
         }
         listModules(hosts[0].textContent);
      } else {
         //if (running === true)
         //   alert('There was a problem with the request.');
      }
   }
}

function listModules(hostname)
{
   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:listModules><ipaddr>' + hostname + '</ipaddr></covise:listModules></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, true, listModulesEnd);  
}

function listModulesEnd(httpRequest)
{

   if (httpRequest.readyState == 4) {
      if (httpRequest.status == 200) {
         var modules = httpRequest.responseXML.documentElement.getElementsByTagName('modules');

         var categoryChoice = document.getElementById('catselect');

         for (var m = 0; m < modules.length; ++m)
         {
            var module = modules[m];
            var category = module.getElementsByTagName('first')[0].textContent;
            var modulename = module.getElementsByTagName('second')[0].textContent;

            if (categories.indexOf(category) == -1)
            {
               var element = document.createElement('option');
               var content = document.createTextNode(category);
               element.appendChild(content);
               categoryChoice.appendChild(element);
               categories.push(category);
               modulesInCategory[category] = new Array;
            }

            modulesInCategory[category].push(modulename);
         }

         updateModuleSelect(categoryChoice.value);

      } else {
         //if (running === true)
         //   alert('There was a problem with the request.');
      }
   }
}

function getRunningModules()
{
   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:getRunningModules></covise:getRunningModules></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, true, getRunningModulesEnd);
}

function getRunningModulesEnd(httpRequest)
{

   if (httpRequest.readyState == 4) {
      if (httpRequest.status == 200) {
         var doc = httpRequest.responseXML.documentElement.firstChild.firstChild;
         var modules = doc.getElementsByTagName('modules');
         for (var module = 0; module < modules.length; ++module)
         {
            addModule(modules[module]);
         }
         getEvent();
      } else {
         //if (running === true)
         //   alert('There was a problem with the request.');
      }
   }
}

function getEvent()
{
   var request = ('<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"> <SOAP-ENV:Body> <covise:getEvent> <uuid xsi:type="xsd:string">' + uuid + '</uuid> <timeout xsi:type="xsd:int">500</timeout></covise:getEvent> </SOAP-ENV:Body></SOAP-ENV:Envelope>');
   sendSOAPRequest(request, true, getEventEnd);
}


function getEventEnd(httpRequest)
{

   if (httpRequest.readyState == 4) {
      if (httpRequest.status == 200) {
         var doc = httpRequest.responseXML.documentElement.firstChild.firstChild;
         var event = doc.getElementsByTagName('event')[0];
         if (event)
         {
            var type = event.getElementsByTagName('type')[0].textContent;
            if (type == "OpenNet")
            {
               var mapname = event.getElementsByTagName('mapname')[0].textContent;
               document.title = "COVISE - " + mapname;
               var table = document.getElementById('controlpanel');
               table.innerHTML = '';
               document.getElementById('filename').value = mapname;
            }
            else if (type == "ModuleAdd")
            {
               var module = event.getElementsByTagName('module')[0];
               addModule(module);
            }
            else if (type == "ModuleDel")
            {
               var id = 'Module_' + event.getElementsByTagName('moduleID')[0].textContent;
               deleteModule(id, false);
            }
            else if (type == "ParameterChange")
            {
               var id = event.getElementsByTagName('moduleID')[0].textContent;
               var parameter = event.getElementsByTagName('parameter')[0];

               var paramName = parameter.getElementsByTagName('name')[0].textContent;
               var paramType = parameter.getElementsByTagName('type')[0].textContent;
               var paramMapped = parameter.getElementsByTagName('mapped')[0].textContent;
               var paramID = id.replace('.', '_', 'g') + '_' + paramName;
               var paramField = document.getElementById(paramID);
               var cell = document.getElementById('Parameter_' + paramID);

               if (cell === null)
               {
                  setTimeout("getEvent()", 10);
                  return;
               }

               var row = cell.parentNode;

               if (paramMapped === "true")
               {
                  if (row.className != 'parameterrow_mapped')
                  {
                     row.className = 'parameterrow_mapped';
                     ++mappedModuleHeaders[id];
                     if (mappedModuleHeaders[id] == 1)
                        setModuleHeaderMapped(id, true);
                  }
               }
               else
               {
                  if (row.className != 'parameterrow')
                  {
                     row.className = 'parameterrow';
                     --mappedModuleHeaders[id];
                     if (mappedModuleHeaders[id] < 1)
                     {
                        setModuleHeaderMapped(id, false);
                        mappedModuleHeaders[id] = 0;
                     }
                  }
               }

               if (paramType.indexOf("Scalar") != -1)
               {
                  paramField.value = parameter.getElementsByTagName('value')[0].textContent;
               }
               else if (paramType.indexOf("FileBrowser") != -1)
               {
                  paramField.value = parameter.getElementsByTagName('value')[0].textContent;
               }
               else if (paramType.indexOf("String") != -1)
               {
                  paramField.value = parameter.getElementsByTagName('value')[0].textContent;
               }
               else if (paramType.indexOf("Vector") != -1)
               {
                  var values = parameter.getElementsByTagName('value');
                  for (var ctr2 = 0; ctr2 < values.length; ++ctr2)
                  {
                     cell.children[ctr2].value = values[ctr2].textContent;
                  }
               }
               else if (paramType.indexOf("ColormapChoice") != -1)
               {
                  var paramValue = parameter.getElementsByTagName('selected')[0].textContent;
                  var colormaps = parameter.getElementsByTagName('colormaps');
                  var cellContent = '<select name="' + paramID + '" id="' + paramID + 
                                    '" size="1" onchange="setColormapChoiceParameter(this.id)"/>';
                  for (var ctr2 = 0; ctr2 < colormaps.length; ++ctr2)
                  {
                     var name = colormaps[ctr2].getElementsByTagName('name')[0].textContent;
                     cellContent += '<option';
                     if (ctr2 == paramValue)
                     {
                        cellContent += ' selected';
                     }
                     cellContent += '>' + name + '</option>';
                  }
                  cellContent += '</select>&nbsp;<img src="colormap/' + paramID + '/' + paramValue + '.png" style="vertical-align:middle; height:16;" />';
                  cell.innerHTML = cellContent;
                  colormapChoices[paramID] = parameter;
               }
               else if (paramType.indexOf("Choice") != -1)
               {
                  var paramValue = parameter.getElementsByTagName('selected')[0].textContent;
                  var choicesNodes = parameter.getElementsByTagName('choices');
                  var cellContent = '<select name="' + paramID + '" id="' + paramID + '" size="1" onchange="setChoiceParameter(this.id)"/>';
                  for (var ctr2 = 0; ctr2 < choicesNodes.length; ++ctr2)
                  {
                     cellContent += '<option';
                     if (ctr2 == paramValue)
                        cellContent += ' selected';
                     cellContent += '>' + choicesNodes[ctr2].textContent + '</option>';
                  }
                  cellContent += '</select>';
                  cell.innerHTML = cellContent;
               }
               else if (paramType.indexOf("Slider") != -1)
               {
                  var value = parameter.getElementsByTagName('value')[0].textContent;
                  var min = parameter.getElementsByTagName('min')[0].textContent;
                  var max = parameter.getElementsByTagName('max')[0].textContent;
                  var slider = sliders[paramID];
                  if (paramType.indexOf("FloatSlider") != -1)
                  {
                     slider.max = parseFloat(max);
                     slider.min = parseFloat(min);
                  }
                  else
                  {
                     slider.max = parseInt(max);
                     slider.min = parseInt(min);
                  }

                  slider.setRealValue(value);
               }
               else if (paramType.indexOf("Boolean") != -1)
               {
                  var paramValue;
                  if (parameter.getElementsByTagName('value')[0].textContent == "true")
                     paramValue = "checked";
                  else
                     paramValue = "";

                  cell.innerHTML = '<input type="checkbox" id="' + paramID + '" value="' + paramID + '" ' + paramValue  + ' " onchange="setBoolParameter(this.id)"/>';
               }
            }
            else
            {
               //alert('Unknown event ' + type);
            }

         } // if (event)

         setTimeout("getEvent()", 10);

      } else {
         //if (running === true)
         //   alert('There was a problem with the request.');
      } // if (httpRequest.status)
   } // if (httpRequest.readyState)

}


function setParameterFromString(parameter, value)
{

   var splits = parameter.split('_');
   var module = splits[0];
   var instance = splits[1];
   var host = splits[2] + '.' + splits[3] + '.' + splits[4] + '.' + splits[5];

   var moduleID = module + '_' + instance + '_' + host;
   var variable = splits[6]
   for (var ctr = 7; ctr < splits.length; ++ctr)
   {
      variable += '_' + splits[ctr];
   }

   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:setParameterFromString><moduleID xsi:type="xsd:string">' + moduleID + '</moduleID><parameter xsi:type="xsd:string">' + variable + '</parameter><value xsi:type="xsd:string">' + value + '</value></covise:setParameterFromString></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, true);

}

function setParameter(parameterID, parameter)
{

   var splits = parameterID.split('_');
   var module = splits[0];
   var instance = splits[1];
   var host = splits[2] + '.' + splits[3] + '.' + splits[4] + '.' + splits[5];

   var moduleID = module + '_' + instance + '_' + host;
   var variable = splits[6]
   for (var ctr = 7; ctr < splits.length; ++ctr)
   {
      variable += '_' + splits[ctr];
   }

   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:setParameter><moduleID xsi:type="xsd:string">' + moduleID + '</moduleID>' + (new XMLSerializer()).serializeToString(parameter).replace('<parameters', '<parameter', '').replace('</parameters', '</parameter', '') + '</covise:setParameter></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, true);

}

function setTextParameter(parameter)
{

   var field = document.getElementById(parameter);
   if (field !== null)
   {
      var value = field.value;
      setParameterFromString(parameter, value);
   }
}

function setVectorParameter(parameter)
{
   var changedField = document.getElementById(parameter);
   if (changedField !== null)
   {
      var value = '';
      var cell = changedField.parentNode;
      for (var ctr = 0; ctr < cell.children.length; ++ctr)
      {
         value += cell.children[ctr].value + ' ';
      }
      value = value.substring(0, value.length - 1);
      setParameterFromString(parameter.substring(0, lastIndexOf('_')), value);
   }
}

function setBoolParameter(parameter)
{

   var field = document.getElementById(parameter);
   if (field !== null)
   {
      var value = (field.checked ? 'true' : 'false');
      setParameterFromString(parameter, value);
   }
}


function setChoiceParameter(parameter)
{

   var field = document.getElementById(parameter);
   if (field !== null)
   {
      // Has to be sent in COVISE native enumeration 1..., thus add one
      var value = field.selectedIndex + 1;
      setParameterFromString(parameter, value);
   }

}

function setColormapChoiceParameter(parameter)
{

   var field = document.getElementById(parameter);
   if (field !== null)
   {
      // Has to be sent in COVISE native enumeration 1..., thus add one
      var value = field.selectedIndex + 1;
      var colormap = colormapChoices[parameter];
      colormap.getElementsByTagName('selected')[0].textContent = value;
      setParameter(parameter, colormap);
   }

}

function openCoviseNet()
{
   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:openNet><filename xsi:type="xsd:string">' + document.form.filename.value + '</filename></covise:openNet></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, true, openCoviseNetEnd);
}

function openCoviseNetEnd(httpRequest)
{

   if (httpRequest.readyState == 4) {
      if (httpRequest.status == 200) {
      //loadControlPanel();
      } else {
         //if (running === true)
         //   alert('There was a problem with the request.');
      }
   }

}

function executeNet()
{
   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:executeNet></covise:executeNet></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, true);
}

function executeModule(module)
{
   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:executeModule><moduleID>' + module + '</moduleID></covise:executeModule></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, true);
}


function addModule(module)
{
   var name = module.getElementsByTagName('name')[0].textContent;
   var category = module.getElementsByTagName('category')[0].textContent;
   var host = module.getElementsByTagName('host')[0].textContent;
   var description = module.getElementsByTagName('description')[0].textContent;
   var instance = module.getElementsByTagName('instance')[0].textContent;
   var id = module.getElementsByTagName('id')[0].textContent;
   var title = module.getElementsByTagName('title')[0].textContent;
   var parameters = module.getElementsByTagName('parameters');

   if (name == 'WebGL' || name == 'Renderer')
   {
      addWebGLRenderer(name, host);
   }

   var table = document.getElementById('controlpanel');
   var row = table.insertRow(-1);
   var cell = row.insertCell(0);

   row.name = 'Module_' + id;
   row.className = "modulesep";
   cell.innerHTML = '<br/>';

   row = table.insertRow(-1);
   row.name = 'Module_' + id;
   row.className = "moduleheader";

   cell = row.insertCell(0);
   cell.innerHTML =
      '<div style="color:black; float:left;">' +
      title + ' (' + name + '_' + instance + '@' + host + ')' +
      '</div>' +
      '<div style="float:right">' +
      '<img src="img/close.png" width="16" height="16" onclick="doDeleteModule(this, true)"/>' +
      '</div>' +
      '<div style="float:right">' +
      '<img src="img/minmax-check.png" width="16" height="16" onclick="hideParameters(this)"/>' +
      '</div>' +
      '<div style="float:right">' +
      '<img src="img/exec.png" width="16" height="16" onclick="doExecuteModule(this)"/>' +
      '</div>';
   cell.colSpan = '2';

   mappedModuleHeaders[id] = 0;

   for (var ctr = 0; ctr < parameters.length; ++ctr)
   {
      var parameter = parameters[ctr];
      var paramName = parameter.getElementsByTagName('name')[0].textContent;
      var paramType = parameter.getElementsByTagName('type')[0].textContent;
      var paramMapped = parameter.getElementsByTagName('mapped')[0].textContent;
      var paramID = id.replace('.', '_', 'g') + '_' + paramName;

      row = table.insertRow(-1);
      row.name = 'Module_' + id;

      if (paramMapped === "true")
      {
         row.className = 'parameterrow_mapped';
         ++mappedModuleHeaders[id];
         if (mappedModuleHeaders[id] == 1)
            setModuleHeaderMapped(id, true);
      }
      else
      {
         row.className = 'parameterrow';
      }

      cell = row.insertCell(0);
      cell.innerHTML = paramName;
      cell = row.insertCell(1);
      cell.id = 'Parameter_' + paramID;
      cell.className = 'parametercell';
      if (paramType.indexOf("Scalar") != -1)
      {
         var paramValue = parameter.getElementsByTagName('value')[0].textContent;
         cell.innerHTML = '<input type="text" size="40" id="' + paramID +
                          '" value="' + paramValue +
                          '" onchange="setTextParameter(this.id)"' + '/>';
      }
      else if (paramType.indexOf("FileBrowser") != -1)
      {
         var paramValue = parameter.getElementsByTagName('value')[0].textContent;
         cell.innerHTML = '<input type="text" size="40" id="' + paramID +
                          '" value="' + paramValue +
                          '" onchange="setTextParameter(this.id)"' + '"/>';
      }
      else if (paramType.indexOf("String") != -1)
      {
         var paramValue = parameter.getElementsByTagName('value')[0].textContent;
         cell.innerHTML = '<input type="text" size="40" id="' + paramID +
                          '" value="' + paramValue +
                          '" onchange="setTextParameter(this.id)"' + '"/>';
      }
      else if (paramType.indexOf("Vector") != -1)
      {
         var elements = parameter.getElementsByTagName('value');
         var size = parseInt(40 / (elements.length + 2));
         var html = "";
         for (var ctr2 = 0; ctr2 < elements.length; ++ctr2)
         {
            html += '<input type="text" size="' + size +
                    '" id="' + paramID + '_' + ctr2 +
                    '" value="' + elements[ctr2].textContent +
                    '" onchange="setVectorParameter(this.id)"' + '"/>';
         }
         cell.innerHTML = html;

      }
      else if (paramType.indexOf("ColormapChoice") != -1)
      {
         var paramValue = parameter.getElementsByTagName('selected')[0].textContent;
         var colormaps = parameter.getElementsByTagName('colormaps');
         var cellContent = '<select name="' + paramID + '" id="' + paramID + 
                           '" size="1" onchange="setColormapChoiceParameter(this.id)"/>';
         for (var ctr2 = 0; ctr2 < colormaps.length; ++ctr2)
         {
            var name = colormaps[ctr2].getElementsByTagName('name')[0].textContent;
            cellContent += '<option';
            if (ctr2 == paramValue)
            {
               cellContent += ' selected';
            }
            cellContent += '>' + name + '</option>';
         }
         cellContent += '</select>&nbsp;<img src="colormap/' + paramID + '/' + paramValue + '.png" style="vertical-align:middle; height:16;" />';
         cell.innerHTML = cellContent;
         colormapChoices[paramID] = parameter;
      }
      else if (paramType.indexOf("Choice") != -1)
      {
         var paramValue = parameter.getElementsByTagName('selected')[0].textContent;
         var choicesNodes = parameter.getElementsByTagName('choices');
         var cellContent = '<select name="' + paramID + '" id="' + paramID + '" size="1" onchange="setChoiceParameter(this.id)"/>';
         for (var ctr2 = 0; ctr2 < choicesNodes.length; ++ctr2)
         {
            cellContent += '<option';
            if (ctr2 == paramValue)
            {
               cellContent += ' selected';
            }
            cellContent += '>' + choicesNodes[ctr2].textContent + '</option>';
         }
         cellContent += '</select>';
         cell.innerHTML = cellContent;
      }
      else if (paramType.indexOf("Slider") != -1)
      {
         var value = parameter.getElementsByTagName('value')[0].textContent;
         var min = parameter.getElementsByTagName('min')[0].textContent;
         var max = parameter.getElementsByTagName('max')[0].textContent;
         var bg = cell.id + '-slider-bg';
         var thumb = cell.id + '-slider-thumb';
         var textfield = cell.id + '-value';
         cell.innerHTML =
            '<div style="position:relative;width:320;"><input  id="' + paramID + '" autocomplete="off" type=text" size="7" value="' +
            value + '"/>' +
            '<div id="' + bg + '" class="yui-h-slider" tabindex="-1" style="background:url(../yui/slider/assets/bg-h.gif) 5px 0 no-repeat; position:absolute; top:0px; left:7em;">' +
            '<div id="' + thumb + '" class="yui-slider-thumb">' +
            '<img src="../yui/slider/assets/thumb-n.gif"></div></div></div>';

         var topConstraint = 0;
         var bottomConstraint = sliderWidth;

         var slider =
            YAHOO.widget.Slider.getHorizSlider(bg, thumb, topConstraint, bottomConstraint);

         slider.animate = true;
         slider.textfield = paramID;

         if (paramType.indexOf("FloatSlider") != -1)
         {
            slider.max = parseFloat(max);
            slider.min = parseFloat(min);
            slider.getRealValue = function() {
               return this.getValue() / sliderWidth *
                  (this.max - this.min) + this.min;
            };
         }
         else
         {
            slider.max = parseInt(max);
            slider.min = parseInt(min);
            slider.getRealValue = function() {
               return Math.round(this.getValue() / sliderWidth *
                                 (this.max - this.min) + this.min);
            };
         }

         slider.setRealValue = function(val) {
            var value = Math.round((sliderWidth / (this.max - this.min) * (val - this.min)));
            if (value < 0)
               value = 0;
            if (value >= sliderWidth)
               value = sliderWidth - 1;
            
            this.setValue(value);
            document.getElementById(this.textfield).value = val;
         };

         slider.subscribe("change", 
                          function(offsetFromStart) {
                             var value = this.getRealValue();
                             if (value < this.min)
                                value = this.min * 1.0;
                             if (value > this.max)
                                value = this.max * 1.0;
                             document.getElementById(this.textfield).value = value;
                          });

         slider.subscribe("slideEnd", 
                          function(offsetFromStart) {
                             var value = this.getRealValue();
                             if (value < this.min)
                                value = this.min * 1.0;
                             if (value > this.max)
                                value = this.max * 1.0;
                             document.getElementById(this.textfield).value = value;
                             if (this.valueChangeSource === 1 )
                                setParameterFromString(this.textfield, value);
                          });

         slider.setRealValue(value);

         sliders[paramID] = slider;

      }
      else if (paramType.indexOf("Boolean") != -1)
      {
         var paramValue;
         if (parameter.getElementsByTagName('value')[0].textContent == "true")
         {
            paramValue = "checked";
         }
         else
         {
            paramValue = "";
         }

         cell.innerHTML = '<input type="checkbox" id="' + paramID + '" value="' + paramID + '" ' + paramValue + ' " onchange="setBoolParameter(this.id)"/>';
      }
      else
         cell.innerHTML = paramType;
   }
}


function deleteModule(id, sendEvent)
{
   if (id.indexOf('Module_WebGL') == 0 || id.indexOf('Module_Renderer') == 0)
   {
      removeWebGLRenderer(id);
   }

   var table = document.getElementById('controlpanel');
   var rows = table.getElementsByTagName("tr");
   for (var ctr = 0; ctr < rows.length;)
   {
      var row = rows[ctr];
      if (row && row.name == id)
         table.firstElementChild.removeChild(row);
      else
         ++ctr;
   }

   if (sendEvent == true)
   {

      var moduleID = id.slice(7); // Slice 'Module_'
      var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"> <SOAP-ENV:Body><covise:deleteModule><moduleID xsi:type="xsd:string">' + moduleID + '</moduleID></covise:deleteModule></SOAP-ENV:Body></SOAP-ENV:Envelope>';
      sendSOAPRequest(request, true);

   }

}

function addWebGLRenderer(modulename, host)
{

   if(modulename == "Renderer")
   {
      var alias = getConfigEntry('System.CRB.ModuleAlias:Renderer/Renderer', 'value');
      if (alias != 'Renderer/WebGL')
         return;
   }

   var port = getConfigEntry('Module.WebGL.Port', 'value');

   if (port == "") port = 32080;
   base = document.location.protocol + "//" + host + ':' + port + "/";

   var webglcell = document.getElementById('WebGLrenderer');
   webglcell.innerHTML = '<br/><iframe src="' + base  + '" width="800" height="700" style="Background:#808080" valign="top"/>';

}

function removeWebGLRenderer(moduleid)
{
   if (moduleid.indexOf('Module_Renderer') == 0)
   {
      var alias = getConfigEntry('System.CRB.ModuleAlias:Renderer/Renderer', 'value');
      if (alias != 'Renderer/WebGL')
         return;
   }

   var webglcell = document.getElementById('WebGLrenderer');
   webglcell.innerHTML = '';
}

function getConfigEntry(section, variable)
{

   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"> <SOAP-ENV:Body><covise:getConfigEntry><section xsi:type="xsd:string">' + section + '</section><variable xsi:type="xsd:string">' + variable + '</variable></covise:getConfigEntry></SOAP-ENV:Body></SOAP-ENV:Envelope>';

   var httpRequest = sendSOAPRequest(request, false);

   return httpRequest.responseXML.getElementsByTagName('value')[0].textContent;

}

function sendSOAPRequest(request, async, callback)
{

   var httpRequest;
   if (window.XMLHttpRequest) { // Mozilla, Safari, ...
      httpRequest = new XMLHttpRequest();
      httpRequest.overrideMimeType('text/xml');
   } else if (window.ActiveXObject) { // IE
      httpRequest = new ActiveXObject("Microsoft.XMLHTTP");
   }

   if (async === true && callback !== undefined && callback !== null)
      httpRequest.onreadystatechange = function() { callback(httpRequest); };

   var base = document.location.protocol + "//" + document.location.host + "/";

   httpRequest.open('POST', base, async);
   httpRequest.send(request);

   return httpRequest;

}


function doDeleteModule(node, fireEvent)
{
   deleteModule(node.parentNode.parentNode.parentNode.name, fireEvent);
}

function doExecuteModule(node)
{
   var id = node.parentNode.parentNode.parentNode.name;
   id = id.slice(7); // Slice 'Module_'
   executeModule(id);
}

function hideParameters(node)
{
   var visibility;
   if (node.src.indexOf('minmax-check') != -1)
   {
      node.src = 'minmax.png';
      visibility = 'collapse';
   }
   else
   {
      node.src = 'minmax-check.png';
      visibility = 'visible';
   }

   var name = node.parentNode.parentNode.parentNode.name;
   var table = document.getElementById('controlpanel');
   var rows = table.getElementsByClassName("parameterrow");
   var row;
   for (row = 0; row < rows.length; ++row)
   {
      if (rows[row] && rows[row].name == name)
         rows[row].style.visibility = visibility;
   }
   rows = table.getElementsByClassName("parameterrow_mapped");
   for (row = 0; row < rows.length; ++row)
   {
      if (rows[row] && rows[row].name == name)
         rows[row].style.visibility = visibility;
   }
}


function updateModuleSelect(forCategory)
{
   var modules = modulesInCategory[forCategory];
   var moduleChoice = document.getElementById('moduleselect');
   moduleChoice.innerHTML = '';

   for (var module = 0; module < modules.length; ++module)
   {
      var element = document.createElement('option');
      var content = document.createTextNode(modules[module]);
      element.appendChild(content);
      moduleChoice.appendChild(element);
   }
}

function instantiateModule()
{
   var module  = document.getElementById('moduleselect').value;
   var host    = document.getElementById('hostselect').value;
   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:instantiateModule><module>' + module + '</module><host>' + host + '</host></covise:instantiateModule></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, false);
}

function controlPanel()
{
   var styleSheets = document.styleSheets;

   for (var ctr = 0; ctr < styleSheets.length; ++ctr)
   {
      if (styleSheets[ctr].title == "modulestyle")
      {
         var rules = styleSheets[ctr].cssRules;

         for (var ctr2 = 0; ctr2 < rules.length; ++ctr2)
         {
            var st = rules[ctr2].selectorText;
            if (st == ".parameterrow" || st == ".moduleheader" || st == ".modulesep")
            {

               if (rules[ctr2].style.display == "none")
                  rules[ctr2].style.display = "inherit";
               else
                  rules[ctr2].style.display = "none";

            }
         }
      }
   }
}

function setModuleHeaderMapped(id, mapped)
{
   var mid = 'Module_' + id;
   var cns = [ 'moduleheader', 'modulesep' ];
   for (var ctr = 0; ctr < cns.length; ++ctr)
   {
      var className = cns[ctr];
      if (mapped)
      {
         var headers = document.getElementsByClassName(className);
         for (var header = 0; header < headers.length; )
         {
            if (headers[header].name == mid)
               headers[header].className = className + '_mapped';
            else
               ++header;
         }
      }
      else
      {
         var headers = document.getElementsByClassName(className + '_mapped');
         for (var header = 0; header < headers.length; )
         {
            if (headers[header].name == mid)
               headers[header].className = className;
            else
               ++header;
         }
      }
   }
}