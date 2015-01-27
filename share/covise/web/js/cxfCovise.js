var uuid = '';
var coviseServiceSoap = new COVISE_ServiceSoap();
//var coviseServiceSoap = new www_hlrs_de_organization_vis_covise_ServiceSoap();
// change url for port number independent
coviseServiceSoap.url = '/COVISE.wsdl';
var recsSimFile = 'net/CoolEmAll/Evaluation3.net';
//var recsSimFile = 'net/CoolEmAll/testCoolEmAll.net';
var serverRoomSimFile = 'net/CoolEmAll/Rechnerraum_test_plmxml.net';
// var serverRoomSimFile = 'net/CoolEmAll/Rechnerraum_test.net';
//var serverRoomSimFile = 'net/RechnerraumOptimierung/Rechnerraum_test-PlmXml.net';


var running = true;

var tabHtmlFile = '';

// sliderLis
var sliderMin = 0;
var sliderMax = 0;

// TODO: maybe better in a config file 
var recsHeight = 0.045;
var serverRoomHeigh = 3.2;

var sliderWidth = 100;
var surfaceSlider = new Object;

// color list
var colormapChoices = new Object;

// html canvas
var canvas = 0;
var canvasCoef = 0;


var sim = '';

// The default logger uses a PopUpAppender 
var log = log4javascript.getDefaultLogger();  

var bCancelUpload = false;        

// create DynaTree
function createDynatree() {
  // Attach the dynatree widget to an existing <div id="tree"> element

  // Initialize the tree 
  $("#tree").dynatree({

    onDblClick: function(node, event) {
      var filePath = node.getKeyPath(false);
      getFileInfoList(filePath);
      node.toggleExpand();
    },  
  
    onCustomRender: function(node) {
      // Render title as columns
      if(node.data.title.indexOf("~") === -1) {
        // Default rendering
        return false;
      }
    
      var cols = node.data.title.split("~");
      var html = '<a class="dynatree-title" href="#">';
      for(var i=0; i<cols.length; i++) {
        html += '<span class="td">' + cols[i] + '</span>';
      }
      return html + "</a>";     
    }
  });

}

// get active path or root path
function getActiveFilePath() {
  var activeNode = $("#tree").dynatree("getActiveNode");
  var filePath ="";

  if (activeNode === null) {
    activeNode = $("#tree").dynatree("getRoot");
    filePath = activeNode.getKeyPath(false);
  } else {
    if (activeNode.data.isFolder) {
      filePath = activeNode.getKeyPath(false) + "/";
    } else {
      filePath = activeNode.getKeyPath(false);   
    }
  }

  return filePath;
}

// get file from input field
function getFile() {
  var fileList = document.getElementById("uploadFile").files;

  if ( typeof FileReader=="undefined" || !fileList) {
    alert("Your browser doesn't support the HTML 5 File API!");
    return;
  }
  if ( fileList.length < 1 )
  {
    alert("Please select a file!");
    return;
  } else {  
    var file = fileList[0];
    return file;
  }
}

// show file info if file changed
function changeFile() {
  $("#divFileProgress").show();

  //FileList Objekt aus dem Input Element mit der ID "fileA"
  var fileList = document.getElementById("uploadFile").files;

  //File Objekt (erstes Element der FileList)
  var file = fileList[0];

  //File Objekt nicht vorhanden = keine Datei ausgewählt oder vom Browser nicht unterstützt
  if(!file)
      return;

  document.getElementById("fileSize").innerHTML = 'Size: ' + file.size + ' B';
  document.getElementById("fileType").innerHTML = 'Type: ' + file.type;
  document.getElementById("progress").value = 0;
  document.getElementById("prozent").innerHTML = "0%";

  $('#uploadHelp').text("Please upload the chosen file now.");
} 


//--------------------- modal new ------------------

/* Button new  */
function clickBtnNew() {
  var activeNode = $("#tree").dynatree("getActiveNode");

  if (activeNode) {
    if (!activeNode.data.isFolder) {
      alert("Please select a folder first!");
      return;  
    }   
  } 
 
  $('#mNew').modal('show');
}

// Button Ok in modal new
// directory will be created under root directory if no node active
function clickBtnOkInMNew() {  
  var filePath = getActiveFilePath();

  var newDir = $("#newDir").val();

  if (!newDir) {
    alert("Please first give the directory name!");
    return;
  } 

  coviseServiceSoap.isDirExist(isDirExistResponse, errorResponse, filePath, newDir); 
}


function isDirExistResponse(response) {
  var result =  response.getResult();

  if(result == true) {
    var isDirExist = response.getIsDirExist();
    if (isDirExist == true) {
      alert("The directory is already exist.");
      return;
    }

    var newDir = $("#newDir").val();
    var filePath = getActiveFilePath();

    coviseServiceSoap.createNewDir(createNewDirResponse, errorResponse, filePath, newDir);
    
  } else {
    log.error("Error during isFileExist().");
  }
}

function createNewDirResponse(response) {
  var result = response.getResult();
  if (result) {
    var filePath = getActiveFilePath();
    getFileInfoList(filePath);

    $("#mNew").modal('hide');    
  } else {
    alert("Error occurred during creation of directory.");
    return;
  }
}


//------------------ modal upload ------------------ 

/* Button upload  */
function clickBtnUpload() {
  var activeNode = $("#tree").dynatree("getActiveNode");

  if (activeNode) {    
    if (!activeNode.data.isFolder) {
      alert("Please select a folder first!");
      return;  
    }   
  }

  changeFile();

  $('#mUpload').modal('show');
}

// Button Upload in modal upload
// File will be uploaded under root directory if no node active
function clickBtnUploadInMUpload(isCancel) {
  var filePath = getActiveFilePath();
  var file = getFile();
  bCancelUpload = isCancel;
  coviseServiceSoap.isFileExist(isFileExistResponse, errorResponse, filePath, file.name);
}

// cancel the upload process
function clickBtnCancelInMUpload(isCancel) {
  bCancelUpload = isCancel;

  var progress = document.getElementById("progress");

  if(progress.value > 0) {   
    var filePath = getActiveFilePath();
    var file = getFile();    
    var uploadFile = filePath + file.name; 
    coviseServiceSoap.deleteDir(cancelUploadResponse, errorResponse, uploadFile);    
  }
}

function cancelUploadResponse(response) {
  var result = response.getResult();
  if (result) {
    var filePath = getActiveFilePath();
    getFileInfoList(filePath);  
  } else {
    alert("Error occurred during deleting of file.");
    return;
  }
}

function isFileExistResponse(response) {
  var result =  response.getResult();

  if(result == true) {
    var isFileExist = response.getIsFileExist();
    if (isFileExist == true) {
      r = confirm("The file is already exist. Are you sure you want to this File?");
      if (r == false) {
        return;  
      }
    }

    var filePath = getActiveFilePath();
    var file = getFile();
    uploadFile(filePath, file);
  } else {
    log.error("Error during isFileExist().");
  }
}

/* Button Ok in modal upload */
function clickBtnOkInMUpload() {
  var progress = document.getElementById("progress");
  if(progress.value < 100) {
    alert("Please first upload the selected file.");
    return;  
  } 
  var filePath = getActiveFilePath();
  
  getFileInfoList(filePath);

  // TODO: not run?!
  //$("#tree").dynatree("getTree").selectKey(activeKey);
  //$("#tree").dynatree("getTree").activateKey(activeKey); 

  $("#mUpload").modal('hide'); 
}

//--------------------- modal delete --------------------
/* Button delete  */
function clickBtnDelete() {
  var activeNode = $("#tree").dynatree("getActiveNode");

  if (!activeNode) {
    alert("Please first select a file or a directory!");
    return; 
  } 

  $('#mDelete').modal('show');
}


/* Button Ok in modal delete */
function clickBtnOkInMDelete() {
  var activeNode = $("#tree").dynatree("getActiveNode");

  var filePath = getActiveFilePath();
  coviseServiceSoap.deleteDir(deleteDirResponse, errorResponse, filePath);
}

function deleteDirResponse(response) {
  var result = response.getResult();
  if (result) {
    var activeNode = $("#tree").dynatree("getActiveNode");

    if (activeNode) {
      activeNode.remove();
    }

    $("#mDelete").modal('hide');    
  } else {
    alert("Error occurred during deleting of file.");
    return;
  }
}

//--------------------- modal csFile --------------------
function clickBtnOkInMCsFiles(btnOkId) {
  var activeNode = $("#tree").dynatree("getActiveNode");

  if (!activeNode || activeNode.data.isFolder) {     
    alert("Please select a file!");
    return;   
  } 

  var paramId = btnOkId.replace("btn", "");

  var activeKey = activeNode.getKeyPath(false);

  var divParamId = "div" + paramId;

  $("#" + divParamId).html(activeKey);
  $("#mCsFiles").modal('hide');

  var coParam = getCoParam(paramId);

  coviseServiceSoap.setParameterFromUploadedFile(setParameterFromUploadedFileResponse, errorResponse, coParam.moduleId, coParam.id, activeKey);      

  resetBtnOkId(btnOkId);
}

function setParameterFromUploadedFileResponse(response) {
  log.info("call setParameterFromUploadedFileResponse()");
}

function changeBtnOkId(paramId) {
  var btnOkId = "btn" + paramId;
  $("#btnOkInMCsFiles").attr("id", btnOkId);
}

function resetBtnOkId(btnOkId) {
  $("#" + btnOkId).attr("id", "btnOkInMCsFiles");
}

function getCoParam(paramId) {
  var coParam = new Object();

  var splits = paramId.split('_');
  var module = splits[0];
  var instance = splits[1];
  var host = splits[2] + '.' + splits[3] + '.' + splits[4] + '.' + splits[5];

  var moduleId = module + '_' + instance + '_' + host;
  var id = splits[6]
  for (var ctr = 7; ctr < splits.length; ++ctr)
  {
    id += '_' + splits[ctr];
  }

  coParam.moduleId = moduleId;
  coParam.id = id;

  return coParam;
}

//get file list from cloud storage directory
function getFileInfoList(path) {
  coviseServiceSoap.getFileInfoList(getFileInfoListRespose, errorResponse, path); 
}

function getFileInfoListRespose(response) {
  var fileInfoList = response.getFileInfoList();

  var curNode = $("#tree").dynatree("getActiveNode");

  if (!curNode) {
    curNode = $("#tree").dynatree("getRoot");
  }

  curNode.removeChildren();

  for (var i = 0; i < fileInfoList.length; ++i)
  {
    if (fileInfoList[i]._fileName !== "." && fileInfoList[i]._fileName !== "..") {
      if (fileInfoList[i]._isDir) {   
        var childNode = curNode.addChild({        
          title: fileInfoList[i]._fileName + '~' + '' + '~' + fileInfoList[i]._fileDate,       
          key: fileInfoList[i]._fileName,
          isFolder: fileInfoList[i]._isDir
        });  
      } else {
        var childNode = curNode.addChild({        
          title: fileInfoList[i]._fileName + '~' + Math.round(fileInfoList[i]._fileSize/100)/10 + '~' + fileInfoList[i]._fileDate,
          key: fileInfoList[i]._fileName,
          isFolder: fileInfoList[i]._isDir
        });
      }
    }
  }

  // Expand all nodes
  $("#tree").dynatree("getRoot").visit(function(node){
    node.expand(true);
  });
}


// upload large file using chunk and base64
/* 
  TODO:
  using chunkid, chunk size, and file size
*/

/*
function uploadFile(uploadDir, file) {
  var progress = document.getElementById("progress");

  var loaded = 0;
  var bytes_per_chunk = 1024*1024; 
  var fileSize = file.size;
  var start = 0;

  var reader = new FileReader();
  var fileName = file.name;

  var fileTrunc = true;

  $('#uploadHelp').text("Uploading ...");

  reader.onprogress = function(evt) { 
    progress.value = (loaded/fileSize) * 100;
    document.getElementById("prozent").innerHTML = Math.round(loaded/fileSize*100) + '%';
  };

  reader.onload = function(evt) { 
    if (evt.loaded) {  

      // value range 0-255, must use Uint8Array 
      var bufView = new Uint8Array(reader.result);
      // string -> 2 byte
      // var str = pack(bufView); 
      // TODO: upload using MTOM
      //coviseServiceSoap.uploadFileMtom(uploadFileResponse, errorResponse, uploadDir, fileName, bufView, fileTrunc, fileSize);      
      
      var str = "";
      for(var i = 0; i < bufView.byteLength; i++)
      {
        // Convert a Unicode number into a character: 65 -> A
        str += String.fromCharCode(bufView[i]);      
      }    
     
      // btoa accepts a “string” where each character represents an 8-bit byte
      var base64str = btoa(str);

      // upload using base64
      coviseServiceSoap.uploadFileBase64(uploadFileResponse, errorResponse, uploadDir, fileName, base64str, fileTrunc, fileSize);

      loaded += evt.loaded;
    
      fileTrunc = false;
      var blob = file.slice(loaded, loaded+bytes_per_chunk);
      reader.readAsArrayBuffer(blob);
    }
  };   

  var blob = file.slice(start, start+bytes_per_chunk);
  reader.readAsArrayBuffer(blob);
}
*/

function uploadFile(uploadDir, file) {
  var progress = document.getElementById("progress");

  var start = 0;
  var chunkSize = 10*1024 * 1024;
  var fileSize = file.size;
  var chunkNr = Math.ceil(file.size/chunkSize);
  var chunkIndex = 0;
  var fileName = file.name;

  var fileTruncated = true;  

  $('#uploadHelp').text("Uploading ...");

  var reader = new FileReader();

  reader.onprogress = function(evt) { 
    progress.value = (start/fileSize) * 100;
    document.getElementById("prozent").innerHTML = Math.round(progress.value) + '%';
  };

  reader.onload = function(evt) { 
    if (bCancelUpload == true) {
      return;  
    }
        
    if((start < fileSize)){
      // value range 0-255, must use Uint8Array 
      var bufView = new Uint8Array(reader.result);

      var str = "";
      for(var i = 0; i < bufView.byteLength; i++)
      {
        // Convert a Unicode number into a character: 65 -> A
        str += String.fromCharCode(bufView[i]);      
      }

      // btoa accepts a “string” where each character represents an 8-bit byte
      var base64str = btoa(str);

      // upload using base64
      coviseServiceSoap.uploadFile(uploadFileResponse, errorResponse, uploadDir, fileName, base64str, chunkIndex, chunkNr, chunkSize, fileSize, fileTruncated);

      start += chunkSize;
      chunkIndex++;
      fileTruncated = false;
      var blob = file.slice(start, start + chunkSize);
      reader.readAsArrayBuffer(blob);
    }
  };   

  var blob = file.slice(start, start + chunkSize);
  reader.readAsArrayBuffer(blob);
}


//TODO: how enable MTOM?
function pack(bytes) {
  var str = "";
  for(var i = 0; i < bytes.length; i += 2) {
    log.info("byte i: " + bytes[i]); 
    var char = bytes[i] << 8;
    log.info("char: " + char);

    if (bytes[i + 1])
        char |= bytes[i + 1];
    str += String.fromCharCode(char);
  }
  return str;
}

function uploadFileResponse(response) {
   log.info("call uploadResourceResponse.");
   var result = response.getResult();
   var lastChunk = response.getLastChunk();
   if (result) {
      if (lastChunk) {
        $('#uploadHelp').text("The upload is successful!"); 
        log.info("The upload process has successful completed!"); 
      } else {
        $('#uploadHelp').text("Uploading ...");
        log.info("uploading file ...");
      }

      // set covise parameter after file uploaded       
   } else {
      $('#uploadHelp').text("Error during file uplaod!");
      log.error("Error during file upload!");
      return;
   }  
}

//TODO: update recs position after windows resized
function canvasCalcOffset()
{ 
   log.info("call canvasCalcOffset.");
   if (canvas) {
      canvas.calcOffset();   
   }  
}

// add event listener, add modules and execute covise
function initCovise(simScenario)
{
   sim = simScenario;
   addCoviseEventListener();
}

function errorResponse(errorNum, errorText)
{
   log.error("call errorResponse: error occurred: " + errorText);
}

// add event listener
function addCoviseEventListener()
{
   log.info("call addCoviseEventListener.");  
   if (uuid === "")
   {
     // uuid = "_requested_"; 
     coviseServiceSoap.addEventListener(addCoviseEventListenerResponse, errorResponse);
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

function addCoviseEventListenerResponse(response)
{  
   log.info("call addCoviseEventListenerResponse.");
   uuid = response.getUuid();
   log.info("uuid: " + uuid);
   getEvent();
}

// get the already running modules
function getCoviseRunningModules()
{
   log.info("call getRunningModules.");
   /* apache cxf does not support inheritance
   coviseServiceSoap.getRunningModules(getRunningModulesResponse, errorResponse);
   */
   
   var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:getRunningModules></covise:getRunningModules></SOAP-ENV:Body></SOAP-ENV:Envelope>';
   sendSOAPRequest(request, true, getCoviseRunningModulesEnd);
}


// used for coolEmAll TAB
function getCoviseRunningModulesEnd(httpRequest)
{
   log.info("call getRunningModulesEnd.");
   if (httpRequest.readyState == 4) {
      if (httpRequest.status == 200) { 
         var doc = httpRequest.responseXML.documentElement.firstChild.firstChild;         
         var modules = doc.getElementsByTagName('modules'); 

         var moduleLength = modules.length;

         if (moduleLength === 0) {   
            openCoviseNet();
         } else {
            var networkFile = httpRequest.responseXML.getElementsByTagName('networkFile')[0].textContent;
            log.info("networkFile: " + networkFile);
            if (networkFile === recsSimFile) {
               sim = "recsSim";
            }

            if (networkFile === serverRoomSimFile) {
               sim = "serverRoomSim";               
            } 

            if (sim === "recsSim" || sim === "serverRoomSim") {
             var tabHtmlFile = sim + '-tab.html';

             $('#tabCoolEmAll a[data-target="#' + sim + '"]').tab('show');      

            
             $('#' + sim).load(tabHtmlFile, function() {
                for (var i = 0; i < moduleLength; ++i) {
                   addModule(modules[i]);
                }          
             });
                          
          }
        }
        
      } 
   }
}


/* apache cxf does not support type inheritance.
   The soap message will be sent.
   */   
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

   // soap url
   var base = document.location.protocol + "//" + document.location.host + "/";

   httpRequest.open('POST', base, async);
   httpRequest.send(request);

   return httpRequest;
}

// open covise a .net file
function openCoviseNet()
{  
   log.info("call openCoviseNet.");
   if (sim === "recsSim") {
      coviseServiceSoap.openNet(openCoviseNetResponse, errorResponse, recsSimFile); 
      log.info("networkFile: " + recsSimFile);  
   } else if (sim === "serverRoomSim") {
      coviseServiceSoap.openNet(openCoviseNetResponse, errorResponse, serverRoomSimFile);
      log.info("networkFile: " + serverRoomSimFile);   
   } else {
      log.info("The simulation scenario " + sim + " does not exist.");
   } 
   
}

function openCoviseNetResponse()
{
  //TODO: error msg for not existing network file 
   log.info("call openCoviseNetResponse.");
   executeNet(executeNetResponse, errorResponse);
}


// execute covise
function executeNet() {
   log.info("call executeNet.");
   coviseServiceSoap.executeNet(executeNetResponse, errorResponse);   
}


function executeNetResponse() {
   log.info("call executeNetResponse.");
}

// execute a covise module
function executeModule(moduleId) {
   log.info("call executeModule.");
   coviseServiceSoap.executeModule(executeModuleResponse, errorResponse, moduleId);   
}


function executeModuleResponse() {
   log.info("call executeModuleResponse.");
}


// get event
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
               var networkFile = event.getElementsByTagName('mapname')[0].textContent;
               document.title = "COVISE - " + networkFile;
            }
            else if (type == "ModuleAdd")
            {
               var module = event.getElementsByTagName('module')[0];
               addModule(module);           
            }
            else if (type == "ModuleDel")
            {
               var module = event.getElementsByTagName('module')[0];
               // delete module
            }
            else if (type == "ParameterChange")
            {  

               var moduleId = event.getElementsByTagName('moduleID')[0].textContent;
               var parameter = event.getElementsByTagName('parameter')[0];
               changeModule(moduleId, parameter);  
            } 
         }
         setTimeout("getEvent()", 10);
       } else {        
          log.info("There was a problem with the request.");
      } 
   } 
}

// stop event listener
function stopCovise()
{
   log.info("call stop.");   
   running = false;   
   if (uuid) {
    removeCoviseEventListener(); 
    uuid = '';
   }   
}

function removeCoviseEventListener()
{  
   log.info("call removeCoviseEventListener");
   coviseServiceSoap.removeEventListener(removeCoviseEventListenerResponse, errorResponse, uuid);     
}


function removeCoviseEventListenerResponse() {
   log.info("call the removeCoviseEventListenerResponse.");
}


// add module
function addModule(module) {
   // sliderMax must be defined before module added
   if (sim === "recsSim") {    
      sliderMax = recsHeight;
      addRecsSimModule(module);
   } else if (sim === "serverRoomSim") {
      sliderMax = serverRoomHeigh;
      addServerRoomSimModule(module);
   } else {
      log.info("There is not module to be added for the sim.: " + sim + ".");
   } 
}


// change module
function changeModule(moduleId, parameter) {
   if (sim === "recsSim") {
      changeRecsSimModule(moduleId, parameter);
   } else if (sim === "serverRoomSim") {
      changeServerRoomSimModule(moduleId, parameter);
   } else {
      log.info("There is not module to be changed for the sim.:: " + sim + ".");
   }
}


// ------------------- recsSim ---------------
// add recsSim module paramters
function addRecsSimModule(module)
{  
   // may be used in the future
   /*
   var moduleName = module.getElementsByTagName('name')[0].textContent;
   var moduleCategory = module.getElementsByTagName('category')[0].textContent;
   var moduleHost = module.getElementsByTagName('host')[0].textContent;
   var moduleDescription = module.getElementsByTagName('description')[0].textContent;
   var moduleInstance = module.getElementsByTagName('instance')[0].textContent;   
   var moduleTitle = module.getElementsByTagName('title')[0].textContent;   
   //*/

   var moduleId = module.getElementsByTagName('id')[0].textContent;  
   log.info("module id: " + moduleId);

   // module CoolEmAllSimulation
   if (moduleId.match("CoolEmAllSimulation") != null) {
      addCoolEmAllSimulation(module);
   }

   // module cutSurface
   if (moduleId.match("CuttingSurface_6") != null) {
      addCutSurface(module);                   
   }
}


// change recsSim module parameter 
function changeRecsSimModule(moduleId, parameter) {
   // module CoolEmAllSimulation
   // changed on 20131108
   // disable this event because of no form elements
   if (moduleId.match("CoolEmAllSimulation") != null) {                  
      //changeCoolEmAllSimulation(moduleId, parameter);
   }

   // module cutSurface
   if (moduleId.match("CuttingSurface_6") != null) {
      changeCutSurface(moduleId, parameter);                   
   }
}


//----------------- serverRoomSim ------------------

// add serverRoomSim module paramters
function addServerRoomSimModule(module)
{
   
   var moduleId = module.getElementsByTagName('id')[0].textContent;  
   log.info("module id: " + moduleId);

   
   // module Rechenraum  
  // if (moduleId.match("SurfaceDemo") != null) {
  //     //addRechenraum(module);
  //     addPlmXmlRechenraum(module);
  // }
  
  
  
  if (moduleId.match("PLMXMLRechenraum") != null) {
      addPlmXmlRechenraum(module);
  }
  
    
   
   // module cutsurface
   if (moduleId.match("CuttingSurface_1") != null) {
      addCutSurface(module);
   }

   // module ColorMap
   if (moduleId.match("Colors_2") != null) {
      addColors(module);
   }
}


// change serverRoomSim module paramters
function changeServerRoomSimModule(moduleId, parameter) {
   // module Rechenraum
   // if (moduleId.match("SurfaceDemo") != null) {
   //    //changeRechenraum(moduleId, parameter);
   //    changePlmXmlRechenraum(moduleId, parameter);
   // }

  
  if (moduleId.match("PLMXMLRechenraum") != null) {
      changePlmXmlRechenraum(moduleId, parameter);
  }
  

   
   // module cutsurface
   if (moduleId.match("CuttingSurface_1") != null) {
      changeCutSurface(moduleId, parameter);
   }

   // module ColorMap
   if (moduleId.match("Colors_2") != null) {
      changeColors(moduleId, parameter)
   }
}


//------------- methods used in all modules ------------

//sent text paramter
function sendTextParameter(paramId)
{

   var field = document.getElementById(paramId);
   if (field !== null)
   {
      var value = field.value;
      sendParameterFromString(paramId, value);
   }
}

// sent string paramter 
function sendParameterFromString(paramId, value)
{
  var coParam = getCoParam(paramId);

  coviseServiceSoap.setParameterFromString(sendParameterFromStringResponse, errorResponse, coParam.moduleId, coParam.id, value);
}


function sendParameterFromStringResponse() {
   log.info("call sendParameterFromStringResponse.");
}


//------------------- module cut/iso-surface -----------------

// add module cut/iso-Surface: slider and the associated input field
function addCutSurface(module) {
    var moduleId = module.getElementsByTagName('id')[0].textContent;
   var parameters = module.getElementsByTagName('parameters'); 
   
   for (var ctr = 0; ctr < parameters.length; ++ctr)
   {
      var parameter = parameters[ctr];
      var paramName = parameter.getElementsByTagName('name')[0].textContent;
      var paramType = parameter.getElementsByTagName('type')[0].textContent; 

      if (paramName.match("scalar") != null) {
         var paramValue = parseFloat(parameter.getElementsByTagName('value')[0].textContent);
         var min = sliderMin;
         var paramMin = Math.round(min*10000)/10000;
         var max = sliderMax;
         var paramMax = Math.round(max*10000)/10000;
   
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName;
         
         var sliderId = paramId + '-slider';
         
         //Dislay Slider         
         var sliderHtml =
            '<input id="' + paramId + '" autocomplete="off" type=text" class="input-mini" value="' +
            Math.round(paramValue*10000)/10000 + '" onchange="setSurfaceSlider(this.id, ' +  paramMax + ', ' +  paramMin + ')" ' +
            'validate="float"> ' +
            '<input type="text" id="' + sliderId +  '" ' + 
            'data-slider-orientation="horizontal" data-slider-selection="after" data-slider-tooltip="hide" ' + 
            '>'; 

          // used in tab recsSim
          if (sim === "recsSim") {
            if ($('#cutSurface').length > 0) {
              $('#cutSurface').replaceWith(sliderHtml);
            }  
          }
          
          // used in tab serverRoomSim
          if (sim === "serverRoomSim") {
            if (!$('#'+paramId).length) {
              $('#divCutSurface').append(sliderHtml);  
            }            
          }          

         // change the input field when the slider is dragged.
         var sliderChange = function() {
            var sliderValue = surfaceSlider.getValue();
            var value = sliderToParamValue(sliderValue, max, min);
            $('#'+paramId).val(value); 
         }; 

         // only send value to covise if event slideStop 
         var sliderStop = function() {
            var sliderValue = surfaceSlider.getValue();
            var value = sliderToParamValue(sliderValue, max, min);
            log.info("sliderValue: " + sliderValue + "; value: " + value);
            sendParameterFromString(paramId, value);
         }; 

         var sliderValue = paramValueToSlider(paramValue, max, min);
       
         surfaceSlider = $('#' + sliderId).slider({                             
                              min: 0,
                              max: sliderWidth,
                              step: 1,
                              value: sliderValue})
                              .on("slide", sliderChange)
                              .on("slideStop", sliderStop)
                              .data('slider');         
      } 
   }      
}


// change parameter type slider 
function changeCutSurface(moduleId, parameter) {
   var paramName = parameter.getElementsByTagName('name')[0].textContent;
   var paramType = parameter.getElementsByTagName('type')[0].textContent; 

   // TODO: why name --> point
   // if (paramName.match("point") != null) {
   if (paramName.match("scalar") != null) {
      var paramValue = parseFloat(parameter.getElementsByTagName('value')[0].textContent);
      var min = sliderMin;
      var max = sliderMax;

      var paramId = moduleId.replace(/\./g, "_") + "_" + paramName;

      var paramField = document.getElementById(paramId);
      if (paramField != null) {
         paramField.value = Math.round(paramValue*10000)/10000;
      } else {
         log.error("There is no field with the id: " + paramId);
      }
      
      surfaceSlider.setValue(paramValueToSlider(paramValue, max, min));      
   }
}

// change slider if the input field value changed
function setSurfaceSlider(id, max, min) {
   var paramValue = parseFloat($('#'+id).val());   
   surfaceSlider.setValue(paramValueToSlider(paramValue, max, min)); 
   // send changed parameter to covise and execute covise
   sendParameterFromString(id, paramValue);
}

// convert slider value to 
function sliderToParamValue(sliderValue, max, min) {
   var sliderCoef = sliderWidth/(max-min);
   var value = Math.round((sliderValue/sliderCoef + min)*10000)/10000;
  // log.info("cut surface slider value by moving slider: " + sliderValue);
  // log.info("cut surface param value: " + value);
   return value;
}

// convert the input field value to slider value
function paramValueToSlider(paramValue, max, min) {
   var sliderCoef = sliderWidth/(max-min);
   var value = Math.round((paramValue - min)*sliderCoef);
  // log.info("cut surface param value by change parameter: " + paramValue);
  // log.info("cut surface slider value: " + value);
   return value;
}


// ---------------------- module Color --------------------

// add color module: select list color
function addColors(module) {
   var moduleId = module.getElementsByTagName('id')[0].textContent;
   var parameters = module.getElementsByTagName('parameters'); 
   
   for (var ctr = 0; ctr < parameters.length; ++ctr)
   {
      var parameter = parameters[ctr];
      var paramName = parameter.getElementsByTagName('name')[0].textContent;
      var paramType = parameter.getElementsByTagName('type')[0].textContent; 

      if (paramType.indexOf("ColormapChoice") != -1) {
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName;

         var colorMapHtml = buildColorMapHtml(paramId, parameter);

         if ($('#colorMap').length > 0) {
             $('#colorMap').replaceWith(colorMapHtml);
         }

         colormapChoices[paramId] = parameter;
      } 
   }
}


// build color select html
function buildColorMapHtml(paramId, parameter) {
   var paramValue = parameter.getElementsByTagName('selected')[0].textContent;
   var colormaps = parameter.getElementsByTagName('colormaps');

   var colorMapHtml = '<select name="' + paramId + '" id="' + paramId + 
                           '" size="1" onchange="sendColormapParameter(this.id)">';

   // set the attribut value?!   
   for (var i = 0; i < colormaps.length; ++i)
   {
      var name = colormaps[i].getElementsByTagName('name')[0].textContent;
      colorMapHtml += '<option';
      if (i == paramValue)
      {
         colorMapHtml += ' selected';
      }
      colorMapHtml += '>' + name + '</option>';
   }

   colorMapHtml += '</select>';

   return colorMapHtml;
}


// change parameter type ColormapChoice im module color
function changeColors(moduleId, parameter) {
   var paramName = parameter.getElementsByTagName('name')[0].textContent;
   var paramType = parameter.getElementsByTagName('type')[0].textContent; 

   if (paramType.indexOf("ColormapChoice") != -1)
   {
        var paramId = moduleId.replace(/\./g, '_') + '_' + paramName;
        
        var colorMapHtml = buildColorMapHtml(paramId, parameter);
        
        var seletedId = '#' + paramId;
        $(seletedId).replaceWith(colorMapHtml);

        // update object colormap
        colormapChoices[paramId] = parameter;
   }
}

// send color parameter
function sendColormapParameter(paramId)
{

   var field = document.getElementById(paramId);
   if (field !== null)
   {
      // Has to be sent in COVISE native enumeration 1..., thus add one
      var value = field.selectedIndex + 1;
      
      var colormap = colormapChoices[paramId];
      colormap.getElementsByTagName('selected')[0].textContent = value;

      var splits = paramId.split('_');
      var module = splits[0];
      var instance = splits[1];
      var host = splits[2] + '.' + splits[3] + '.' + splits[4] + '.' + splits[5];

      var moduleId = module + '_' + instance + '_' + host;
      var id = splits[6]
      for (var ctr = 7; ctr < splits.length; ++ctr)
      {
         id += '_' + splits[ctr];
      }

      var request = '<?xml version="1.0" encoding="UTF-8"?><SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:covise="http://www.hlrs.de/organization/vis/covise"><SOAP-ENV:Body><covise:setParameter><moduleID xsi:type="xsd:string">' + moduleId + '</moduleID>' + (new XMLSerializer()).serializeToString(colormap).replace('<parameters', '<parameter', '').replace('</parameters', '</parameter', '') + '</covise:setParameter></SOAP-ENV:Body></SOAP-ENV:Envelope>';
      sendSOAPRequest(request, true);  

      // comment out on 20131128: execute covise by click button now
      // executeNet();        
   }
}



// --------------- module CoolEmAllSimulation ---------------

// add module CoolEmAllSimulation on 20131108
function addCoolEmAllSimulation(module) {
   var moduleId = module.getElementsByTagName('id')[0].textContent;
   var parameters = module.getElementsByTagName('parameters'); 
   
   for (var ctr = 0; ctr < parameters.length; ++ctr)
   {
      var parameter = parameters[ctr];
      var paramName = parameter.getElementsByTagName('name')[0].textContent;
      var paramType = parameter.getElementsByTagName('type')[0].textContent;

      if (paramName.match("DEBBURL") != null) { 
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName; 
        // var paramValue = parameter.getElementsByTagName('value')[0].textContent;
        
         if ($('#debbUrlDiv').length) {            
            //$('#debbUrlDiv').attr("id", paramId);
            var val = locationVars('path_plmxml'); 
            //$('#debbUrlDiv').append(val); 
            //$('#debbUrlDiv').html(val);
            $('#debbUrlDiv').append('<input type="text" class="input-mini" placeholder="' + val + '" disabled>');          
            sendParameterFromString(paramId, val);            
         }
      }

      if (paramName.match("ExperimentID") != null) { 
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName; 

         if ($('#experimentDiv').length) {
            var val = locationVars('experiment');
            $('#experimentDiv').append('<input type="text" class="input-mini" placeholder="' + val + '" disabled>'); 
            sendParameterFromString(paramId, val);
         }           
      }

      if (paramName.match("TrialID") != null) { 
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName; 

         if ($('#trialDiv').length) {
            var val = locationVars('trial');
            $('#trialDiv').append('<input type="text" class="input-mini" placeholder="' + val + '" disabled>'); 
            sendParameterFromString(paramId, val);
         }           
      }

      if (paramName.match("ExperimentType") != null) { 
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName; 

         if ($('#experimentTypeDiv').length) {
            var val = locationVars('experiment_type');
            $('#experimentTypeDiv').append('<input type="text" class="input-mini" placeholder="' + val + '" disabled>'); 
            sendParameterFromString(paramId, val);
         }           
      }

      if (paramName.match("DatabasePath") != null) { 
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName; 

         if ($('#objectPathDiv').length) {
            var val = locationVars('object_path');
            $('#objectPathDiv').append('<input type="text" class="input-mini" placeholder="' + val + '" disabled>');
            var databasePath = 'sim/' + locationVars('experiment') + '/' + locationVars('trial') +'/' + val;
            sendParameterFromString(paramId, databasePath);
         }           
      }

      if (paramName.match("ToSimulate") != null) { 
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName; 

         if ($('#testbedInstanceDiv').length) {
            var val = locationVars('testbed_instance');
            $('#testbedInstanceDiv').append('<input type="text" class="input-mini" placeholder="' + val + '" disabled>');
            sendParameterFromString(paramId, val);
         }           
      } 

      if (paramName.match("StartTime") != null) { 
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName;

         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName; 

         if ($('#startTimeDiv').length) {
            var val = locationVars('startTime');
            $('#startTimeDiv').append('<input type="text" class="input-mini" placeholder="' + val + '" disabled>'); 
            sendParameterFromString(paramId, val);
         }              
      } 

      if (paramName.match("EndTime") != null) { 
         var paramId = moduleId.replace(/\./g, "_") + "_" + paramName; 

         if ($('#endTimeDiv').length) {
            var val = locationVars('endTime');
            $('#endTimeDiv').append('<input type="text" class="input-mini" placeholder="' + val + '" disabled>');
            sendParameterFromString(paramId, val);
         }           
      } 

   }

   if ($('#runSim').length) {
      $('#runSim').attr("id", moduleId); 
   }
}

// read symfony2 parameters from cookie 
function getCookies() {
   document.cookie = 'username=ayi'
   $.cookie('the_cookie', 'the_value');
   var value = $.cookie('the_cookie');
   var usr = $.cookie('username');  
   alert("setted cookie read : " + value);
   alert("cookie username: " + usr);
}

// read symfony2 parameters from url
function locationVars (vr)
{  
  /* 
   var src = String( window.location.href ).split('?')[1];  
   var vrs = src.split('&');
  
   for (var x = 0, c = vrs.length; x < c; x++)
   {
    if (vrs[x].indexOf(vr) != -1)
    {
      return decodeURI( vrs[x].split('=')[1] );
      break;
    };
   };   
   */
   return 'test';
};


//change paramter in module CoolEmAllSimulation
function changeCoolEmAllSimulation(moduleId, parameter) {  
   var paramName = parameter.getElementsByTagName('name')[0].textContent;
   var paramType = parameter.getElementsByTagName('type')[0].textContent; 
   var paramId = moduleId.replace(/\./g, '_') + '_' + paramName;
   if (paramType.indexOf("FileBrowser") != -1)
   {
      var paramField = document.getElementById(paramId);
      paramField.value = parameter.getElementsByTagName('value')[0].textContent;                 
   }

   if (paramName.match("project_dir") != null) { 
      var paramField = document.getElementById(paramId);
      paramField.value = parameter.getElementsByTagName('value')[0].textContent; 
   }  
}


//----------------- module Rechenraum --------------------

// add module rechenraum in canvas
function addRechenraum(module) {
   buildRechenraumHtml();
   
   var moduleId = module.getElementsByTagName('id')[0].textContent;
   var parameters = module.getElementsByTagName('parameters');

   var posRacks = new Array();
   var sizeRacks = new Array();
   var flowrateRacks = new Array();
  
   for (var ctr = 0; ctr < parameters.length; ++ctr)
   {
      var parameter = parameters[ctr];
      var paramName = parameter.getElementsByTagName('name')[0].textContent;
      var paramType = parameter.getElementsByTagName('type')[0].textContent;
   
      if (paramType.indexOf("Vector") != -1) {
         if (paramName.match("model_size") != null) { 
            var values = parameter.getElementsByTagName('value');
            var sizeModel = new Array();
            for (var i = 0; i < values.length; ++i)
            {
               sizeModel[i] = values[i].textContent;                 
            } 
            // set canvas size 
            setCanvas(sizeModel);
         }
         else if (paramName.match("size_rack_") != null) {
            // array indexes start with 0, but racks with 1 
            var index = paramName.split("_")[2]-1;
            var values = parameter.getElementsByTagName('value');

            if (values.length == 3) {
               sizeRacks[index] = vectorToArray(values);
            }
            else {
               log.error("Error in rack size of module Rechenraum.");
               return;
            }
         }
         else if (paramName.match("pos_rack_") != null ) {
            // array indexes start with 0, but racks with 1
            var index = (paramName.split("_"))[2]-1;
            var values = parameter.getElementsByTagName('value');

            if (values.length == 3) {
               posRacks[index] = vectorToArray(values);
            }
            else {
               log.error("Error in rack pos. of module Rechenraum.");
               return;
            }
         }             
      } 
      else if (paramType.indexOf("Scalar") != -1) {
         // array indexes start with 0, but racks with 1
         var index = (paramName.split("_"))[2]-1;
         if (paramName.match("flowrate_rack_") != null ) {
            flowrateRacks[index] = parameter.getElementsByTagName('value')[0].textContent;
         }
      }
   }   
   
   var racks = {
      size: sizeRacks, 
      pos: posRacks, 
      flowrate: flowrateRacks
   };

   // draw rect on canvas
   drawRects(racks, moduleId);
   
}


//----------------- module PlmXmlRechenraum --------------------

// add module rechenraum in canvas
function addPlmXmlRechenraum(module) {
   buildRechenraumHtml();
   
   var moduleId = module.getElementsByTagName('id')[0].textContent;
   var parameters = module.getElementsByTagName('parameters');

   var posRacks = new Array();
   var sizeRacks = new Array();
   var flowrateRacks = new Array();
  
   for (var ctr = 0; ctr < parameters.length; ++ctr)
   {
      var parameter = parameters[ctr];
      var paramName = parameter.getElementsByTagName('name')[0].textContent;
      var paramType = parameter.getElementsByTagName('type')[0].textContent;    

      // plmxml file
      if (paramType.indexOf("FileBrowser") != -1)
      //if (paramType.indexOf("String") != -1)
      {        
        // change the related id
        if (paramName.match("PLMXML_Path") != null) {
        //  if (paramName.match("BCFile") != null) {
          var paramId = moduleId.replace(/\./g, "_") + "_" + paramName;
          var paramValue = parameter.getElementsByTagName('value')[0].textContent;    

          // change button id
          if ($('#plmXmlFile').length) {         
            $('#plmXmlFile').attr("id", paramId); 
          } 

          // change div id
          if ($('#divPlmXmlFile').length) { 
            var divPlmXmlFileId = "div" + paramId; 
            $('#divPlmXmlFile').attr("id", divPlmXmlFileId); 
            $("#" + divPlmXmlFileId).html(paramValue);          
            
          }
        }        
      }

      if (paramType.indexOf("Vector") != -1) {
         if (paramName.match("model_size") != null) { 
            var values = parameter.getElementsByTagName('value');
            var sizeModel = new Array();
            for (var i = 0; i < values.length; ++i)
            {
               sizeModel[i] = values[i].textContent;                 
            } 
            // set canvas size 
            setCanvas(sizeModel);
         }
         else if (paramName.match("size_rack_") != null) {
            // array indexes start with 0, but racks with 1 
            var index = paramName.split("_")[2]-1;
            var values = parameter.getElementsByTagName('value');

            if (values.length == 3) {
               sizeRacks[index] = vectorToArray(values);
            }
            else {
               log.error("Error in rack size of module Rechenraum.");
               return;
            }
         }
         else if (paramName.match("pos_rack_") != null ) {
            // array indexes start with 0, but racks with 1
            var index = (paramName.split("_"))[2]-1;
            var values = parameter.getElementsByTagName('value');

            if (values.length == 3) {
               posRacks[index] = vectorToArray(values);
            }
            else {
               log.error("Error in rack pos. of module Rechenraum.");
               return;
            }
         }             
      } 
      else if (paramType.indexOf("Scalar") != -1) {
         // array indexes start with 0, but racks with 1
         var index = (paramName.split("_"))[2]-1;
         if (paramName.match("flowrate_rack_") != null ) {
            flowrateRacks[index] = parameter.getElementsByTagName('value')[0].textContent;
         }
      }
   }   
   
   var racks = {
      size: sizeRacks, 
      pos: posRacks, 
      flowrate: flowrateRacks
   };

   // draw rect on canvas
   drawRects(racks, moduleId);
   
}


// build html for rechenraum module
function buildRechenraumHtml() {
   var fieldsetId = $('#moduleParam');

   var fieldsetHtmlBegin = '<fieldset> ' + 
                           '<legend>Computer Room</legend> ';

   var fieldsetHtmlEnd = '</fieldset> ';

   var rowHtmlBegin = '<div class="row-fluid"> ';

   var rowHtmlEnd = ' </div>';   

   var span5HtmlBegin = '<div class="span5"> ' +
                        '<div class="control-group"> ';

   var span2HtmlBegin = '<div class="span2"> ' +
                        '<div class="control-group"> ';

   var spanHtmlEnd = '</div> ' + 
                     '</div>';

   var canvasHtml =  '<div class="span12" id="computerRoom"> ' + 
                     '<canvas id="canvas"> ' +                  
                     '</canvas> ' +                             
                     '</div> '; 

   var posHtml = span5HtmlBegin + '<label> Rack Pos</label> ' + buildRackHtml("pRack", 3) + spanHtmlEnd;
   var sizeHtml = span5HtmlBegin + '<label> Rack Size</label> ' + buildRackHtml("sRack", 3) + spanHtmlEnd;
   var flowrateHtml = span2HtmlBegin + '<label> Rack Flowrate</label> ' + buildRackHtml("fRack", 1) + spanHtmlEnd;
   
   if (fieldsetId.length > 0) {
      if (!$('#computerRoom').length) {
         fieldsetId.append(fieldsetHtmlBegin + 
                     rowHtmlBegin +
                     posHtml +
                     sizeHtml +
                     flowrateHtml +
                     rowHtmlEnd +
                     rowHtmlBegin +
                     canvasHtml +
                     rowHtmlEnd +
                     fieldsetHtmlEnd);   
      }      
   }   
}


// vector to array
function vectorToArray(values) {
   len = values.length;

   var a = new Array(len);

   for (var i = 0; i < len; ++i)
   {
      a[i] = values[i].textContent;                 
   }  

   return a;
}



function sendRackParameterResponse() {
   log.info("call sendRackParamResponse.");
}


// send rack parameter to covise
function sendRackParameter(moduleId, paramId, value)
{  
   // send to COVISE
   coviseServiceSoap.setParameterFromString(sendRackParameterResponse, errorResponse, moduleId, paramId, value);

   // send changed parameter to covise and execute covise
   // comment out on 20131128: execute covise by click button now
   // executeNet();      
}


// build rack input html
function buildRackHtml(prop, len) {
   var html = '';
   if (len == 1) {
      html += '<input type="text" id="' + prop + '" class="input-mini"> '
   } else {
      for (var i = 0; i < len; ++i)
      {
        html += '<input type="text" id="' + prop + i + '" class="input-mini"> '
      } 
   }
   return html;
}
        

// HTML5 Canvas
function setCanvas(sModel) {
   if (sModel.length != 3) {
      log.error("Error during set canvas.");
      return;
   }

   canvas = new fabric.Canvas('canvas');

   // canvas size depend on windows
   var canvasWidth = $("#computerRoom").width(); 

   canvasCoef = (Math.floor((canvasWidth/sModel[0])*10))/10; 

   canvas.backgroundColor = "lightGray";

   canvas.setWidth(canvasWidth);
   canvas.setHeight(sModel[1]*canvasCoef); 
}
//change size of Canvas
function changeCanvas(sModel){
  if(sModel.length != 3){
    log.error("Error during set canvas!");
    return;
  }
  var canvasWidth = $("#computerRoom").width();

  canvasCoef = (Math.floor((canvasWidth/sModel[0])*10))/10;
  canvas.setWidth(canvasWidth);
  canvas.setHeight(sModel[1]*canvasCoef);
}

// created a new class rect with label
var LabeledRect = fabric.util.createClass(fabric.Rect, {

  type: 'labeledRect',

  initialize: function(element, options) {
      this.callSuper('initialize', element, options);
      options && this.set('lockUniScaling', options.lockUniScaling);
      options && this.set('label', options.label || '');
  },

  toObject: function() {
            return fabric.util.object.extend(this.callSuper('toObject'), {
                label: this.label,
                lockUniScaling: this.lockUniScaling
            });
        },

   _render: function(ctx) {
            this.callSuper('_render', ctx);
            ctx.font = '20px Times';
            ctx.fillStyle = 'white';
            ctx.fillText(this.label, -this.width / 2 + 4, -this.height / 2 + 20);
        }
});


/*
   Fabric.js: Left/Top values of all objects (including groups) correspond to an object's center, 
   and are relative to top/left corner of a canvas. 
   Left/Top values of objects in groups are relative to groups' centers
*/
function drawRects(racks, moduleId) {
   var rects = new Array();
   var canvasHeight = canvas.getHeight();

   var rectWidth = 0;
   var rectHeight = 0;
   var rectLeft = 0;
   var rectTop = 0;

   for (var i=0; i < racks.pos.length; ++i)      
   {
      // origin: Rack: left/top in renderer, rect: left/bottom
    
      if (racks.size[i][0]) {
         rectWidth = getRectWidthByRackSizeX(racks.size[i][0]); 
      } else {
         rectWidth = 0;
         log.info(i + "rack size x is not corret.");
      }

      if (racks.size[i][1]) {
         rectHeight = getRectHeigthByRackSizeY(racks.size[i][1]);
      } else {  
         rectHeight = 0;  
         log.info(i + "rack size y is not corret.");  
      }

      if (!racks.size[i][2]) {
         racks.size[i][2] = 0;
         log.info(i + "rack size z is not corret."); 
      }


      if (racks.pos[i][0]) {
        rectLeft = getRectLeftByRackPosX(racks.pos[i][0]);         
      } else {
         rectLeft = 0;
         log.info(i + "rack pos x is not corret.");
         
      }

      
      if (racks.pos[i][1]) {
        rectTop = getRectTopByRackPosY(racks.pos[i][1]);
      } else {
         rectTop = canvasHeight;
         log.info(i + "rack pos y not correct.");
      }
      

      if (!racks.pos[i][2]) {
         racks.pos[i][2] = 0;
         log.info(i + "rack pos z not correct.");
      }


      if (!racks.flowrate[i]) {
         racks.flowrate[i] = 0;
         log.info(i + "rack flowrate not correct.");
      }

      
      // create a rect with label for each rack
      var rackId = i+1;
      rects[i] = new LabeledRect({
                     left: rectLeft,
                     top:  rectTop,                     
                     width: rectWidth,
                     height: rectHeight, 
                     originX: "left", 
                     originY: "bottom",
                     label: rackId,
                     fill: 'blue'                                      
                  });
      
      /*
      // create a rect for each rack
      rects[i]= new fabric.Rect({
                     left: rectLeft,
                     top:  rectTop,                     
                     width: rectWidth,
                     height: rectHeight, 
                     originX: "left", 
                     originY: "bottom"                                       
                  });
      //*/

      // set rect id
      rects[i].set("id", i); 
    
    
      // set posZ, sizeZ and flowrate
      rects[i].set("rackPosZ", racks.pos[i][2]); 
      rects[i].set("rackSizeZ", racks.size[i][2]);
      rects[i].set("rackFlowrate", racks.flowrate[i]);
      // set module id
      rects[i].set("coviseModuleId", moduleId);

      rects[i].lockRotation = true; 
      rects[i].cornerSize = 6;
      rects[i].cornerColor = "white";
      // add rect into canvas
      canvas.add(rects[i]); 
      
   }


   canvas.on({      
      'mouse:down': function(options) {
         updateMouseDown(options)},
      'object:selected': function(options){
         updateSelected(options)},
      'object:moving': function(options){
         updateMoving(options)},
      'object:scaling': function(options){
         updateScaling(options)},
      'object:modified': function(options){
         updateModified(options)}
      });

    canvas.on("after:render", function(){canvas.calcOffset();});

  // canvas.calcOffset();
}


 /*
   The event object:moving" (or "object:scaling") are fired continuously 
   every time an object is moved (or scaled) even by one pixel. 
   On the other hand, events like "object:modified" are fired only 
   at the end of the action (object modification).
 */ 
// event handler for mouse down
function updateMouseDown(options) {   
   if (!options.target) {
      disableRackInputFields();
   }
} 

// event handler for object selected
function updateSelected(options) {  
   var rectId = -1;
   if (options.target) {        
      var thisTarget = options.target;            
      if (thisTarget.isType('group')) { 
            thisTarget.lockRotation = true; 
            disableRackInputFields();
         } else {    
         enableRackInputFields();
         setRectOnchangeRackInputFields(thisTarget);
         setRackSizeInputFields(thisTarget);
         setRackPosInputFields(thisTarget);
         setRackFlowrateInputField(thisTarget);
      }
   }
} 

// event handler for object moving
function updateMoving(options) {   
   var rectId = -1;
   if (options.target) {      
      var thisTarget = options.target;
      // modified on 20140121 due to rect with label         
      //if (thisTarget.isType('rect')) { 
        if (thisTarget.isType('labeledRect')) {         
         setRackPosInputFields(thisTarget);
      }
   }
} 

// show rack pos in input fields 
function setRackPosInputFields(rect) {
   // text fields for rack pos.

   var pRack0 = document.getElementById("pRack0");         
   var pRack1 = document.getElementById("pRack1");
   var pRack2 = document.getElementById("pRack2");

   var rectLeft = rect.getLeft();
   var rectTop = rect.getTop();  

   var canvasHeight = canvas.getHeight();

   var rackPosX = getRackPosXByRect(rectLeft);
   var rackPosY = getRackPosYByRect(canvasHeight, rectTop);
   var rackPosZ = rect.get("rackPosZ");

   // show rack pos
   pRack0.value = Math.round(rackPosX*100)/100;
   pRack1.value = Math.round(rackPosY*100)/100;         
   pRack2.value = Math.round(rackPosZ*100)/100;    
}



// event handler for object scaling
function updateScaling(options) {   
   var rectId = -1;
   if (options.target) {    
      var thisTarget = options.target;
      
      // modified on 20140121 due to rect with label         
      //if (thisTarget.isType('rect')) { 
        if (thisTarget.isType('labeledRect')) {         
         setRackSizeInputFields(thisTarget);
         setRackPosInputFields(thisTarget);       
      }
   }
}

// set rack size in input fields 
function setRackSizeInputFields(rect) {
   // rack size
   var sRack0 = document.getElementById("sRack0");         
   var sRack1 = document.getElementById("sRack1");
   var sRack2 = document.getElementById("sRack2");

   var rectWidth = rect.getWidth();
   var rectHeight = rect.getHeight();

   // rect size
   var rackSizeX = getRackSizeXByRect(rectWidth);
   var rackSizeY = getRackSizeYByRect(rectHeight); 
   var rackSizeZ = rect.get("rackSizeZ");

   // show rack size
   sRack0.value = Math.round(rackSizeX*100)/100;  
   sRack1.value = Math.round(rackSizeY*100)/100;
   sRack2.value = Math.round(rackSizeZ*100)/100;
}


// set rect flowrate field
function setRackFlowrateInputField(rect) {  
   // flowrate
   var fRack = document.getElementById("fRack");
   var rackFlowrate = rect.get("rackFlowrate");

   // display flowrate
   if (rect.get("rackFlowrate")) {
      fRack.value = Math.round(rackFlowrate*100)/100;
   } else {
      fRack.value = 0; 
   } 
}

  
// event handler for rack input fields changed
function setRectOnchangeRackInputFields(rect) {
   // rack size
   var sRack0 = document.getElementById("sRack0");         
   var sRack1 = document.getElementById("sRack1");
   var sRack2 = document.getElementById("sRack2");

   // rack pos.
   var pRack0 = document.getElementById("pRack0");         
   var pRack1 = document.getElementById("pRack1");
   var pRack2 = document.getElementById("pRack2");

   // flowrate
   var fRack = document.getElementById("fRack");

   var moduleId = rect.get("coviseModuleId");
   var rackId = rect.get("id") + 1;
   var rackPosParamId = "pos_rack_" + rackId;
   var rackSizeParamId = "size_rack_" + rackId;
   var rackFlowrateParamId = "flowrate_rack_" + rackId;

   // TODO: show error to user (browser)
   // rack size
   sRack0.onchange = function() {
      // update rect width
      if (parseFloat(this.value) !== "NaN") {
         var rackSizeX = parseFloat(this.value);
         var rectWidth = getRectWidthByRackSizeX(rackSizeX);

         rect.set("width", rectWidth);          
         rect.setCoords();       
         canvas.renderAll();

         // send changed param to covise 
         var rackSizeValue = sRack0.value + " " + sRack1.value + " " + sRack2.value;         
         sendRackParameter(moduleId, rackSizeParamId, rackSizeValue);

      } else {
         log.error("The given rack size X length value must be a number.");     
      }
   };

   sRack1.onchange = function() {
       if (parseFloat(this.value) !== "NaN") {
         var rackSizeY = parseFloat(this.value);
         var rectHeight = getRectHeigthByRackSizeY(rackSizeY);        

         rect.set("height", rectHeight); 
      
         rect.setCoords();              
         canvas.renderAll();

         // send changed param to covise
         var rackSizeValue = sRack0.value + " " + sRack1.value + " " + sRack2.value;         
         sendRackParameter(moduleId, rackSizeParamId, rackSizeValue);         
      } else {
         log.error("The given rack size Y width value must be a number.");     
      }
      
   };

   sRack2.onchange = function() {     
      // only set rack because of rect 2D
      if (parseFloat(this.value) !== "NaN") {
         var rackSizeZ = parseFloat(this.value);

         // only update rackSize         
         rect.set("rackSizeZ", rackSizeZ);         

         // send changed param to covise
         var rackSizeValue = sRack0.value + " " + sRack1.value + " " + sRack2.value;         
         sendRackParameter(moduleId, rackSizeParamId, rackSizeValue);
      } else {
          log.error("The given rack size height value must be a number.");   
      }
   };

   // rack pos.
   pRack0.onchange = function() {
      if (parseFloat(this.value) !== "NaN") {
         var rackPosX = parseFloat(this.value);        
         var rectLeft = getRectLeftByRackPosX(rackPosX);

         rect.set("left", rectLeft); 
         rect.setCoords();        
         canvas.renderAll();

         // send changed param to covise 
         var rackPosValue = pRack0.value + " " + pRack1.value + " " + pRack2.value;         
         sendRackParameter(moduleId, rackPosParamId, rackPosValue);
      } else {
         log.error("The given rack coordinates left value must be a number."); 
      }
   };
   
   pRack1.onchange = function() {      
      if (parseFloat(this.value) !== "NaN") {        
         var rackPosY = parseFloat(this.value);
         var rectTop = getRectTopByRackPosY(rackPosY);
        
         rect.set("top", rectTop);
         rect.setCoords();        
         canvas.renderAll();

         // send changed param to covise
         var rackPosValue = pRack0.value + " " + pRack1.value + " " + pRack2.value;         
         sendRackParameter(moduleId, rackPosParamId, rackPosValue);
      } else {
         log.error("The given rack coordinates bottom value must be a number.");
      }      
   }; 

   pRack2.onchange = function() {
      if (parseFloat(this.value) !== "NaN") {
         var rackPosZ = parseFloat(this.value);
        
         rect.set("rackPosZ", rackPosZ);
         
         // send changed param to covise
        var rackPosValue = pRack0.value + " " + pRack1.value + " " + pRack2.value;         
         sendRackParameter(moduleId, rackPosParamId, rackPosValue);
      } else {
         log.error("The given rack pos height value must be a number.");
      }      
   }; 


   // flowrate
   fRack.onchange = function() { 
      if (parseFloat(this.value) !== "NaN") {
         var rackFlowrate = parseFloat(this.value);

         // only set rack
         rect.set("rackFlowrate", rackFlowrate);
   
         // send changed param to covise
         sendRackParameter(moduleId, rackFlowrateParamId, rackFlowrate);
      } else {
      log.error("flowrate must be a number");   
      }
   };
}


// event handler for object modified
function updateModified(options) {
   if (options.target) {      
      var thisTarget = options.target; 
      setRackOnModifiedRect(thisTarget);
   }
}

// set rack properties 
function setRackOnModifiedRect(thisTarget) {
   var rectTop = 0;
   var rectLeft = 0;
   var rectWidth = 0;
   var rectHeight = 0;
 
   // The objects in group are all positioned relative to the center of the group. 
   // The scale value of all object is always 1 
   // Therefore setting group scale value 1 before sent parameter value to covise
   if (thisTarget.isType('group')) { 
      var groupLeft = thisTarget.getLeft();
      var groupTop = thisTarget.getTop();
      var groupScaleX = thisTarget.getScaleX();
      var groupScaleY = thisTarget.getScaleY();

      log.info("------------- group before setting ---------------");
      log.info("group scaleX/scaleY: " + thisTarget.getScaleX() + "/" + thisTarget.getScaleY());
      log.info("group scaleX/scaleY by .prop: " + thisTarget.scaleX + "/" + thisTarget.scaleY);
      log.info("group top/left by getProp(): " + thisTarget.getTop() + "/" + thisTarget.getLeft());
      log.info("group top/left by .prop: " + thisTarget.top + "/" + thisTarget.left);   
      log.info("group top/left by get(prop): " + thisTarget.get("top") + "/" + thisTarget.get("left"));
      log.info("group width/height by getProp(): " + thisTarget.getWidth() + "/" + thisTarget.getHeight());
      log.info("group width/height by .prop: " + thisTarget.width + "/" + thisTarget.height);
      log.info("group width/height get(prop): " + thisTarget.get("width") + "/" + thisTarget.get("height"));
      
     
      thisTarget.forEachObject(function(object) {
        // modified on 20140121 due to rect with label
        // if (object.isType('rect')) {
          if (object.isType('labeledRect')) {

            log.info("############ object in group before setting ################");
            log.info("rect id: " + object.get("id"));
            log.info("object scale scaleX/scaleY: " + object.getScaleX() + "/" + object.getScaleY());
            log.info("object scale scaleX/scaleY by .prop: " + object.scaleX + "/" + object.scaleY);
            log.info("rect top/left by getProp(): " + object.getTop() + "/" + object.getLeft());
            log.info("rect top/left by .prop: " + object.top + "/" + object.left);   
            log.info("rect top/left by get(prop): " + object.get("top") + "/" + object.get("left"));
            log.info("rect width/height by getProp(): " + object.getWidth() + "/" + object.getHeight());
            log.info("rect width/height by .prop: " + object.width + "/" + object.height);
            log.info("rect width/height get(prop): " + object.get("width") + "/" + object.get("height"));

            
            var rectTopToCenterPoint = object.getTop()*groupScaleY;
            var rectLeftToCenterPoint = object.getLeft()*groupScaleX;

            var rectWidth = object.getWidth()*groupScaleX;
            var rectHeight = object.getHeight()*groupScaleY;

            var rectTop = groupTop + rectTopToCenterPoint;        
            var rectLeft = groupLeft + rectLeftToCenterPoint;
            
            // set object in the group 
            object.set("top", rectTopToCenterPoint);
            object.set("left", rectLeftToCenterPoint);
            object.set("width", rectWidth);
            object.set("height", rectHeight);
            object.scale(1);

            log.info("xxxxxxxxxxxxxxxxx object in group after setting xxxxxxxxxxxxxxxxxxx");
            log.info("object scale scaleX/scaleY: " + object.getScaleX() + "/" + object.getScaleY());
            log.info("object scale scaleX/scaleY by .prop: " + object.scaleX + "/" + object.scaleY);
            log.info("rect top/left by getProp(): " + object.getTop() + "/" + object.getLeft());
            log.info("rect top/left by .prop: " + object.top + "/" + object.left);   
            log.info("rect top/left by get(prop): " + object.get("top") + "/" + object.get("left"));
            log.info("rect width/height by getProp(): " + object.getWidth() + "/" + object.getHeight());
            log.info("rect width/height by .prop: " + object.width + "/" + object.height);
            log.info("rect width/height get(prop): " + object.get("width") + "/" + object.get("height"));
            
            setRackParameter(object, rectTop, rectLeft);
         } else {
            log.error("The object is not rect!");
            return;
         }
      }); 

      // set group scale to 1
      thisTarget.set("width", thisTarget.getWidth());
      thisTarget.set("height", thisTarget.getHeight());
      thisTarget.scale(1);

      /*
      log.info("****************** group after setting *******************");
      log.info("group scaleX/scaleY: " + thisTarget.getScaleX() + "/" + thisTarget.getScaleY());
      log.info("group scaleX/scaleY by .prop: " + thisTarget.scaleX + "/" + thisTarget.scaleY);
      log.info("group top/left by getProp(): " + thisTarget.getTop() + "/" + thisTarget.getLeft());
      log.info("group top/left by .prop: " + thisTarget.top + "/" + thisTarget.left);   
      log.info("group top/left by get(prop): " + thisTarget.get("top") + "/" + thisTarget.get("left"));
      log.info("group width/height by getProp(): " + thisTarget.getWidth() + "/" + thisTarget.getHeight());
      log.info("group width/height by .prop: " + thisTarget.width + "/" + thisTarget.height);
      log.info("group width/height get(prop): " + thisTarget.get("width") + "/" + thisTarget.get("height"));
      */
   } else {
      // changed on 20140121 due to rect with label
      //if (thisTarget.isType('rect')) {
        if (thisTarget.isType('labeledRect')) {

         log.info("################# object before setting #################");
         log.info("rect id: " + thisTarget.get("id"));
         log.info("scale scaleX/scaleY: " + thisTarget.getScaleX() + "/" + thisTarget.getScaleY());
         
         var rectTop = thisTarget.getTop();
         var rectLeft = thisTarget.getLeft();
         var rectWidth = thisTarget.getWidth();
         var rectHeight = thisTarget.getHeight();

         /*
         log.info("rect top/left by getProp(): " + rectTop + "/" + rectLeft);
         log.info("rect top/left by .prop: " + thisTarget.top + "/" + thisTarget.left);   
         log.info("rect top/left by get(prop): " + thisTarget.get("top") + "/" + thisTarget.get("left"));
         log.info("rect width/height by getProp(): " + rectWidth + "/" + rectHeight);
         log.info("thisTarget width/height by .prop: " + thisTarget.width + "/" + thisTarget.height);   
         log.info("rect width/height by get(prop): " + thisTarget.get("width") + "/" + thisTarget.get("height"));
         */      

         // set scale to 1 for modified object
         thisTarget.set("width", rectWidth);      
         thisTarget.set("height", rectHeight);
         thisTarget.scale(1);

         /*
         log.info("xxxxxxxxxxxxxxxxx object after setting xxxxxxxxxxxxxxxxx");
         log.info("scale scaleX/scaleY: " + thisTarget.getScaleX() + "/" + thisTarget.getScaleY());
         log.info("rect top/left by getProp(): " + thisTarget.getTop() + "/" + thisTarget.getLeft());
         log.info("rect top/left by .prop: " + thisTarget.top + "/" + thisTarget.left);
         log.info("rect top/left by .prop: " + thisTarget.top + "/" + thisTarget.left);   
         log.info("rect top/left by get(prop): " + thisTarget.get("top") + "/" + thisTarget.get("left"));
         log.info("rect width/height by getProp(): " + thisTarget.getWidth() + "/" + thisTarget.getHeight());
         log.info("rect width/height by .prop: " + thisTarget.width + "/" + thisTarget.height);                
         log.info("rect width/height by get(prop): " + thisTarget.get("width") + "/" + thisTarget.get("height"));
         */ 

         setRackParameter(thisTarget, rectTop, rectLeft);         
      } else {
         log.error("The object is not rect.");
         return;
      }      
   }
}


// set rack pos. and size, used for rect and rect in group
function setRackParameter(rect, rectTop, rectLeft) {
   var rectWidth = rect.getWidth();
   var rectHeight = rect.getHeight();

   // rack: top/left, rect: center
   var rackSizeX = getRackSizeXByRect(rectWidth);
   var rackSizeY = getRackSizeYByRect(rectHeight);
   var rackSizeZ = rect.get("rackSizeZ");
   

   var canvasHeight = canvas.getHeight();

   var rackPosX = getRackPosXByRect(rectLeft);
   var rackPosY = getRackPosYByRect(canvasHeight, rectTop);
   var rackPosZ = rect.get("rackPosZ");

   log.info("rack size x/y/z: " + rackSizeX + "/" + rackSizeY + "/" + rackSizeZ);
   log.info("rack pos x/y/z: " + rackPosX + "/" + rackPosY + "/" + rackPosZ);   

   // set param and send to covise
   var moduleId = rect.get("coviseModuleId");
   var rackId = rect.get("id") + 1;

   var rackSizeParamId = "size_rack" + "_" + rackId;
   var rackPosParamId = "pos_rack" + "_" + rackId;   

   var rackSizeValue = rackSizeX + " " + rackSizeY + " " + rackSizeZ;
   var rackPosValue = rackPosX + " " + rackPosY + " " + rackPosZ;

   sendRackParameter(moduleId, rackSizeParamId, rackSizeValue);
   sendRackParameter(moduleId, rackPosParamId, rackPosValue);
}

// disable rack input fields
function disableRackInputFields() {
    // rack size
   var sRack0 = document.getElementById("sRack0");         
   var sRack1 = document.getElementById("sRack1");
   var sRack2 = document.getElementById("sRack2");
   sRack0.value = " "; 
   sRack0.disabled = true; 
   sRack1.value = " "; 
   sRack1.disabled = true; 
   sRack2.value = " "; 
   sRack2.disabled = true; 

   // rack pos.
   var pRack0 = document.getElementById("pRack0");         
   var pRack1 = document.getElementById("pRack1");
   var pRack2 = document.getElementById("pRack2");
   pRack0.value = " "; 
   pRack0.disabled = true; 
   pRack1.value = " "; 
   pRack1.disabled = true; 
   pRack2.value = " "; 
   pRack2.disabled = true;

   // flowrate
   var fRack = document.getElementById("fRack");
   fRack.disabled = true; 
   fRack.value = " ";
}

// enable rack input fields
function enableRackInputFields() {
    // rack size
   var sRack0 = document.getElementById("sRack0");         
   var sRack1 = document.getElementById("sRack1");
   var sRack2 = document.getElementById("sRack2");
   sRack0.disabled = false; 
   sRack1.disabled = false;
   sRack2.disabled = false;

   // rack pos.
   var pRack0 = document.getElementById("pRack0");         
   var pRack1 = document.getElementById("pRack1");
   var pRack2 = document.getElementById("pRack2");
   pRack0.disabled = false; 
   pRack1.disabled = false;
   pRack2.disabled = false;

   // flowrate
   var fRack = document.getElementById("fRack");
   fRack.disabled = false;
}

// get rect left if rack pos x changed
function getRectLeftByRackPosX(rackPosX) {
   var rectLeft = parseFloat(rackPosX)*canvasCoef;
   return rectLeft;
}


// get rect top if rack pos y changed
function getRectTopByRackPosY(rackPosY) {
   var canvasHeight = canvas.getHeight();
   var rectTop = canvasHeight - parseFloat(rackPosY)*canvasCoef;
   return rectTop;
}

// get rect width if rack size x changed
function getRectWidthByRackSizeX(rackSizeX) {
   var rectWidth = parseFloat(rackSizeX)*canvasCoef;
   return rectWidth;
}

// get rect height if rack size y changed
function getRectHeigthByRackSizeY(rackSizeY) {
   var rectHeight = parseFloat(rackSizeY)*canvasCoef;
   return rectHeight;
}

// get rack pos x if rect modified
function getRackPosXByRect(rectLeft) {
   var rackPosX = rectLeft/canvasCoef;
   return rackPosX;
}

// get rack pos y if rect modified
function getRackPosYByRect(canvasHeight, rectTop) {
   var rackPosY = (canvasHeight - rectTop)/canvasCoef;
   return rackPosY;
}

// get rack size x if rect modified
function getRackSizeXByRect(rectWidth) {
   var rackSizeX = rectWidth/canvasCoef;
   return rackSizeX;
}

// get rack size y if rect modified
function getRackSizeYByRect(rectHeight) {
   var rackSizeY = rectHeight/canvasCoef;
   return rackSizeY
}

// get rect using id
function getRectById(index) {
   var rects = canvas.getObjects();
   var rect;
   for (var i = 0; i < rects.length; i++) {
      if(rects[i].id == index) {              
         //gets the object with index
         rect = rects[i];
         break;
      }
   } 
   return rect;
}


// update module rechenraum during event change parameter 
function changeRechenraum(moduleId, parameter) {
   var paramName = parameter.getElementsByTagName('name')[0].textContent;
   var paramType = parameter.getElementsByTagName('type')[0].textContent;

   if (paramName.match("_rack_") !== null) {     
      // array indexes start with 0, but racks with 1 
      var index = paramName.split("_")[2]-1;
      var rect = getRectById(index);

      if (rect === null) {
         log.error("rect with id: " + index + " does not exist.");
         return;
      }
              
      if (paramType.indexOf("Vector") != -1) {
         var values = parameter.getElementsByTagName('value');
         var rackParamValue = vectorToArray(values);

          if (paramName.match("pos_rack_") != null) {         
            updateRectByChangeRackPos(rect, rackParamValue);
         }

         if (paramName.match("size_rack_") != null) {
            updateRectByChangeRackSize(rect, rackParamValue);
         }
      }
   

      if (paramType.indexOf("Scalar")) {
         if (paramName.match("flowrate_rack_") != null) {
            var rackParamValue = parameter.getElementsByTagName('value')[0].textContent;
            updateRectByChangeRackFlowrate(rect, rackParamValue);
         }
      }
   }
}


// update module lmXmlRechenraum during event change parameter 
function changePlmXmlRechenraum(moduleId, parameter) {
   var paramName = parameter.getElementsByTagName('name')[0].textContent;
   var paramType = parameter.getElementsByTagName('type')[0].textContent;

   if (paramName.match("PLMXML_Path") != null) {
    //if (paramName.match("BCFile") != null) {  
      var paramId = moduleId.replace(/\./g, "_") + "_" + paramName;
      var paramValue = parameter.getElementsByTagName('value')[0].textContent;    

      var divPlmXmlFileId = "div" + paramId;
      $("#" + divPlmXmlFileId).html(paramValue);
   }
   if (paramName.match("model_size") != null) { 
            var values = parameter.getElementsByTagName('value');
            var sizeModel = new Array();
            for (var i = 0; i < values.length; ++i)
            {
               sizeModel[i] = values[i].textContent;                 
            } 
            // set canvas size 
            changeCanvas(sizeModel);
         }

   if (paramName.match("_rack_") !== null) {     
      // array indexes start with 0, but racks with 1 
      var index = paramName.split("_")[2]-1;
      var rect = getRectById(index);

      if (rect === null) {
         log.error("rect with id: " + index + " does not exist.");
         return;
      }
              
      if (paramType.indexOf("Vector") != -1) {
         var values = parameter.getElementsByTagName('value');
         var rackParamValue = vectorToArray(values);

          if (paramName.match("pos_rack_") != null) {         
            updateRectByChangeRackPos(rect, rackParamValue);
         }

         if (paramName.match("size_rack_") != null) {
            updateRectByChangeRackSize(rect, rackParamValue);
         }
      }
   

      if (paramType.indexOf("Scalar")) {
         if (paramName.match("flowrate_rack_") != null) {
            var rackParamValue = parameter.getElementsByTagName('value')[0].textContent;
            updateRectByChangeRackFlowrate(rect, rackParamValue);
         }
      }
   }
}


// update rect during event change change parameter 
function updateRectByChangeRackPos(rect, rackPos) {
   var group = canvas.getActiveGroup(); 

   if (group) {      
      var isRectInActiveGroup = isObjectInActiveGroup(group, rect); 
      if (isRectInActiveGroup) {
         // update rect in selected group
         updateRectInActiveGroupByChangeRackPos(group, rect, rackPos)  
      } else {
         // update not selected rect
         updateNonActiveRectByChangeRackPos(rect, rackPos);
      }      
   } else {
      var activeObj = canvas.getActiveObject();

      if (activeObj) {
         var isActiveObj = isActiveObject(rect, activeObj);

         if (isActiveObj) {
            //update selected rect
            updateActiveRectByChangeRackPos(rect, rackPos);
         } else {
            // update not selected rect
            updateNonActiveRectByChangeRackPos(rect, rackPos);
         }          
      } else {
         updateNonActiveRectByChangeRackPos(rect, rackPos);      
      }
   }
}

 // update rect during event change parameter 
function updateRectByChangeRackSize(rect, rackSize) {
   var group = canvas.getActiveGroup();

   if (group) {      
      var isRectInActiveGroup = isObjectInActiveGroup(group, rect); 
      if (isRectInActiveGroup) {
         // update rect in selected group
         updateRectInActiveGroupByChangeRackSize(group, rect, rackSize)  
      } else {
         // update not selected rect
         updateNonActiveRectByChangeRackSize(rect, rackSize);
      }      
   } else {
      var activeObj = canvas.getActiveObject();

      if (activeObj) {
         var isActiveObj = isActiveObject(rect, activeObj);

         if (isActiveObj) {
            //update selected rect
            updateActiveRectByChangeRackSize(rect, rackSize);
         } else {
            // update not selected rect
            updateNonActiveRectByChangeRackSize(rect, rackSize);
         }        
      } else {
          updateNonActiveRectByChangeRackSize(rect, rackSize);      
      }
  }
}

// check if object in seleted group
function isObjectInActiveGroup(group, rect) {
   var isInActiveGroup = false;

   group.forEachObject(function(object) {
      if (rect.id == object.id) {
         isInActiveGroup = true;
      }
   });

   return isInActiveGroup;
}

 // check if object selected
function isActiveObject(rect, activeObject) {
   var isActiveObj = false;

   if (rect.id == activeObject.id) {
      isActiveObj = true;
   }
  
   return isActiveObj;
}


// update rect in selected group during event change parameter 
function updateRectInActiveGroupByChangeRackPos(group, rect, rackPos) { 
   group.removeWithUpdate(rect);
   var rectLeft = getRectLeftByRackPosX(rackPos[0]);
   var rectTop = getRectTopByRackPosY(rackPos[1]);

   rect.set("top", rectTop);
   rect.set("left", rectLeft);
   rect.set("rackPosZ", rackPos[2]);
   rect.setCoords();

   group.addWithUpdate(rect);

   canvas.renderAll();
}

// update rect in selected group during event change parameter 
function updateRectInActiveGroupByChangeRackSize(group, rect, rackSize) {
   group.removeWithUpdate(rect);
   var rectWidth = getRectWidthByRackSizeX(rackSize[0]);
   var rectHeight = getRectHeigthByRackSizeY(rackSize[1]);

   rect.set("width", rectWidth);
   rect.set("height", rectHeight);
   rect.set("rackSizeZ", rackSize[2]);
   rect.setCoords();

   group.addWithUpdate(rect);
   
   canvas.renderAll();
}

// update selected rect during event change parameter 
function updateActiveRectByChangeRackPos(rect, rackPos) {  
   // rack input field 
   var pRack0 = document.getElementById("pRack0");         
   var pRack1 = document.getElementById("pRack1");
   var pRack2 = document.getElementById("pRack2");

   // set reck input field
   var rectLeft = 0;
   var rackPosX = Math.round(rackPos[0]*100)/100;
   if (pRack0.value != rackPosX) {
      pRack0.value = rackPosX;
      rectLeft = getRectLeftByRackPosX(rackPos[0]);
      rect.set("left", rectLeft).setCoords();
   }

   var rectTop = 0;
   var rackPosY = Math.round(rackPos[1]*100)/100;
   if (pRack1.value != rackPosY) {
      pRack1.value = rackPosY;
      rectTop = getRectTopByRackPosY(rackPos[1]);
      rect.set("top", rectTop).setCoords();
   }


   var rackPosZ = Math.round(rackPos[2]*100)/100;
   if (pRack2.value != rackPosZ) {
      pRack2.value = rackPosZ; 
      // change rect rackPosZ value
      rect.set("rackPosZ", rackPos[2]);
   } 

   rect.setCoords();    
   canvas.renderAll();   
}

// utpdate not seleted rect during event change parameter 
function updateNonActiveRectByChangeRackPos(rect, rackPos) {
   var rectLeft = getRectLeftByRackPosX(rackPos[0]);
   var rectTop = getRectTopByRackPosY(rackPos[1]);       

   rect.set("top", rectTop);
   rect.set("left", rectLeft);
   rect.set("rackPosZ", rackPos[2]);

   rect.setCoords();
   canvas.renderAll();  
}


// update selected rect during event change parameter 
function updateActiveRectByChangeRackSize(rect, rackSize) {
      
   // set input field 
   var sRack0 = document.getElementById("sRack0");         
   var sRack1 = document.getElementById("sRack1");
   var sRack2 = document.getElementById("sRack2");

   var rackSizeX = Math.round(rackSize[0]*100)/100;
   if (sRack0.value != rackSizeX) {
      sRack0.value = rackSizeX; 
      var rectWidth = getRectWidthByRackSizeX(rackSize[0]);       
      rect.set("width", rectWidth);
   } 

   var rackSizeY = Math.round(rackSize[1]*100)/100;
   if (sRack1.value != rackSizeY.value) {
      sRack1.value = rackSizeY;
      var rectHeight = getRectHeigthByRackSizeY(rackSize[1]);  
      rect.set("height", rectHeight);
   } 

   var rackSizeZ = Math.round(rackSize[2]*100)/100;
   if (sRack2.value != rackSizeZ) {
      sRack2.value = Math.round(rackSize[2]*100)/100;
      rect.set("rackSizeZ", rackSize[2]); 
   } 

   rect.setCoords();
   canvas.renderAll();  
}


// update not selected rect during event change parameter 
function updateNonActiveRectByChangeRackSize(rect, rackSize) {

   var rectWidth = getRectWidthByRackSizeX(rackSize[0]);
   var rectHeight = getRectHeigthByRackSizeY(rackSize[1]);
   
   // update rect top/left 
   rect.set("width", rectWidth);
   rect.set("height", rectHeight);
   rect.set("rackSizeZ", rackSize[2]);

   rect.setCoords();
   canvas.renderAll();   
}


// update rect prop. rack flowrate 
function updateRectByChangeRackFlowrate(rect, rackFlowrate) {
   var activeObj = canvas.getActiveObject();

   if (activeObj) {
      var isActiveObj = isActiveObject(rect, activeObj);

      if (isActiveObj) {
         // update rack input field for selected rect
         var fRack = document.getElementById("fRack");
         var rackFlowrate = Math.round(rackFlowrate*100)/100;    

         if (rackFlowrate != fRack.value) {
            log.info("rack flowrate in rect: " + rect.get("rackFlowrate"));            
            fRack.value = rackFlowrate;
         } 
      }          
   }

   rect.set("rackFlowrate", rackFlowrate);
   log.info("rack flowrate updated in rect: " + rect.get("rackFlowrate")); 
}


















