var slider;
var sliderWidth = 200;

var objects = { };          // COVISE data objects
var interactors = { };      // matrices for placement of interactors

var tMin = 0; var tMax = 0; // timestep indices
var timestep = 1;           // current timestep                      

var request;                // XMLHttpRequest objects
var viewRequest;
var interactorRequest;

var colorShader;
var colorFlatShader;
var colorPickShader;
var textureShader;
var textureFlatShader;
var currentShader;

var gl;

var mouseDown = false;

var last = $V([0, 0]);  // last mouse press location

// translation and rotation of modelview matrix
var trans = Matrix.Translation($V([0, 0, -20])).ensure4x4();
var rot = Matrix.Diagonal([1.0, 1.0, 1.0, 1.0]);
var view = trans.x(rot);

var viewIndex = -1; // viewer position number for master/slave
var pickIndex = 1;

var mvMatrix;     // modelview matrix
var pMatrix;      // perspective matrix
var sphere = { }; // sphere interactor geometry
var quad = { };   // quad label geometry
var logo = { };   // logo label geometry

/* compile a shader 
 *     id: script id in DOM
 */
function getShader(id)
{
   var shaderScript = document.getElementById(id);
   if (!shaderScript)
      return null;
   
   var str = "";
   var k = shaderScript.firstChild;
   while (k) {
      if (k.nodeType == 3)
         str += k.textContent;
      k = k.nextSibling;
   }
   
   var shader;
   if (shaderScript.type == "x-shader/x-fragment")
      shader = gl.createShader(gl.FRAGMENT_SHADER);
   else if (shaderScript.type == "x-shader/x-vertex")
      shader = gl.createShader(gl.VERTEX_SHADER);
   else
      return null;
   
   gl.shaderSource(shader, str);
   gl.compileShader(shader);
   
   if (!gl.getShaderi(shader, gl.COMPILE_STATUS)) {
      alert(gl.getShaderInfoLog(shader));
      return null;
   }
   return shader;
}

/* select a shader
 *    shader: the shader to use
 */
function useShader(shader)
{
   currentShader = shader;
   gl.useProgram(shader);
}

/* initialize shader programs */
function initShaders()
{
   var colorFragmentShader = getShader("color-shader-fs");
   var colorFlatFragmentShader = getShader("color-flat-shader-fs");
   var colorPickFragmentShader = getShader("color-pick-shader-fs");
   
   var colorVertexShader = getShader("color-shader-vs");
   var colorFlatVertexShader = getShader("color-flat-shader-vs");
   
   var textureFragmentShader = getShader("texture-shader-fs");
   var textureVertexShader = getShader("texture-shader-vs");
   
   var textureFlatFragmentShader = getShader("texture-flat-shader-fs");
   
   colorShader = gl.createProgram();
   gl.attachShader(colorShader, colorVertexShader);
   gl.attachShader(colorShader, colorFragmentShader);
   
   gl.linkProgram(colorShader);
   if (!gl.getProgrami(colorShader, gl.LINK_STATUS))
      alert("Could not initialise color shaders");
   
   colorFlatShader = gl.createProgram();
   gl.attachShader(colorFlatShader, colorFlatVertexShader);
   gl.attachShader(colorFlatShader, colorFlatFragmentShader);

   gl.linkProgram(colorFlatShader);
   if (!gl.getProgrami(colorFlatShader, gl.LINK_STATUS))
      alert("Could not initialise color flat shaders");

   colorPickShader = gl.createProgram();
   gl.attachShader(colorPickShader, colorFlatVertexShader);
   gl.attachShader(colorPickShader, colorPickFragmentShader);

   gl.linkProgram(colorPickShader);
   if (!gl.getProgrami(colorPickShader, gl.LINK_STATUS))
      alert("Could not initialise color pick shaders");

   textureShader = gl.createProgram();
   gl.attachShader(textureShader, textureVertexShader);
   gl.attachShader(textureShader, textureFragmentShader);

   gl.linkProgram(textureShader);
   if (!gl.getProgrami(textureShader, gl.LINK_STATUS))
      alert("Could not initialise texture shaders");

   textureFlatShader = gl.createProgram();
   gl.attachShader(textureFlatShader, textureVertexShader);
   gl.attachShader(textureFlatShader, textureFlatFragmentShader);

   gl.linkProgram(textureFlatShader);
   if (!gl.getProgrami(textureFlatShader, gl.LINK_STATUS))
      alert("Could not initialise textureflat shaders");

   useShader(colorShader);
}

function webSocket()
{
   var ws = new WebSocket("ws://" + document.location.hostname + ":32082");

   ws.onopen = function() {
      ws.send("Hello");
   };    
   ws.onmessage = function (e) {

      var doc = (new DOMParser()).parseFromString(e.data, "text/xml").documentElement;
      tMin = doc.getAttribute("tMin");
      tMax = doc.getAttribute("tMax");
      
      objs = doc.getElementsByTagName('obj');
      var t0 = (new Date).getTime();
      for (var i = 0; i < objs.length; i++)
         eval(objs[i].firstChild.data);
      
      if (objs.length > 0) {
         createBuffers();
         var t2 = (new Date).getTime();
         //console.log("time: %d %d", t2 - t0);
         draw();
      }
   };
   ws.onclose = function() {
      alert("websocket closed");
   };
}

/* AJAX request for COVISE objects */
function getData()
{
   if (!request && window.XMLHttpRequest)
      request = new XMLHttpRequest();

   if (request) {
      var names = "";
      for (var object in objects)
         names += objects[object]["name"] + ",";

      //request.abort();
      url = "/getdata?objects=" + names;
      //url = "/stream?objects=" + names;

      request.onreadystatechange = gotData;
      request.open("GET", url, true);
      request.send(null);
   }
}

/* AJAX callback for master/slave viewer positions */
function gotView()
{
   /*
     if (viewRequest.readyState == 4) {
     var doc = viewRequest.responseXML.documentElement;
     viewIndex = doc.getAttribute("index");
     if (doc.hasAttribute("view")) {
     eval(doc.getAttribute("view"));
     draw();
     }
     }
   */
}

/* AJAX callback for COVISE objects */
function gotData()
{
   if (request.readyState == 4) {
 
      var doc = request.responseXML.documentElement;
      tMin = doc.getAttribute("tMin");
      tMax = doc.getAttribute("tMax");
      /*
        var v = doc.getAttribute("view");

        if (v != viewIndex) {
        if (!viewRequest && window.XMLHttpRequest)
        viewRequest = new XMLHttpRequest();

        var url = "/view"
        viewRequest.abort();               
        viewRequest.onreadystatechange = gotView;
        viewRequest.open("GET", url, true);
        viewRequest.send(null);
            
        v = viewIndex;
        }
      */
      objs = request.responseXML.getElementsByTagName('obj');
      var t0 = (new Date).getTime();
      for (var i = 0; i < objs.length; i++)
         eval(objs[i].firstChild.data);

      if (objs.length > 0) {
         createBuffers();
         var t2 = (new Date).getTime();
         //console.log("time: %d %d", t2 - t0);
         draw();
      }

      setTimeout("getData()", 0);
   }
}

/* create WebGL buffers for COVISE objects from WebGL arrays */
function createBuffers()
{
   for (var object in objects) {
      
      // create buffers if they don't exist
      if (!objects[object].buffers) {
         objects[object].buffers = { };
         objects[object].buffers.vertices = gl.createBuffer();
         objects[object].buffers.indices = gl.createBuffer();
         //console.log("  triangles: : %d", objects[object]["indices"].length / 3);
         gl.bindBuffer(gl.ARRAY_BUFFER, objects[object].buffers.vertices);
         gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(objects[object]["vertices"]), gl.STATIC_DRAW);
         gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

         gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, objects[object].buffers.indices);
         gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new WebGLUnsignedShortArray(objects[object]["indices"]), gl.STATIC_DRAW);

         if (typeof objects[object]["colors"] == "undefined" && objects[object]["type"] != "lines") {
            var size = objects[object]["vertices"].length;
            objects[object]["colors"] = new Array(size * 4);
            for (var i = 0; i < size; i ++) {
               objects[object]["colors"].push(0.2);
               objects[object]["colors"].push(0.2);
               objects[object]["colors"].push(0.2);
               objects[object]["colors"].push(0.0);
            }
         }

         if (typeof objects[object]["colors"] != "undefined") {
            objects[object].buffers.colors = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, objects[object].buffers.colors);
            gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(objects[object]["colors"]), gl.STATIC_DRAW);
         }

         if (typeof objects[object]["normals"] != "undefined") {
            objects[object].buffers.normals = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, objects[object].buffers.normals);
            gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(objects[object]["normals"]), gl.STATIC_DRAW);
         }

         if (typeof objects[object]["texcoords"] != "undefined") {
            objects[object].buffers.texcoord = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, objects[object].buffers.texcoord);
            gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(objects[object]["texcoords"]), gl.STATIC_DRAW);
         }
      }
   }
}

/* delete objects, interactors, colormaps
 *    name: the name of the object that is no longer used
 *    TODO: free WebGL buffers
 */
function deleteObject(name)
{

   if (objects[name] && typeof objects[name].interactors != "undefined")
      for (var i = 0; i < objects[name].interactors.length; i ++) {
         delete interactors[objects[name].interactors[i]];
         deleteInteractorCheckbox("checkbox" + (objects[name].interactors[i]));
      }

   delete interactors[name];
   deleteInteractorCheckbox("checkbox" + name);

   delete objects[name];

   var end = name.indexOf("(");
   if (end != -1) {
      name = name.substring(0, end);
   }
   var t = document.getElementById('colormaps');
   var element = document.getElementById("colorMap" + name);
   if (element)
      t.deleteRow(element.rowIndex);
}

/* add a colormap to the colormap table
 *    name: name of the module
 *    filename: png image url on the server
 *    min, max: minimum and maximum value of the colormap
 */
function addColorMap(name, filename, min, max)
{
   var end = name.indexOf("(");
   if (end != -1) {
      name = name.substring(0, end);
   }
   var t = document.getElementById('colormaps');
   var element = document.getElementById("colorMap" + name);
   if (element)
      t.deleteRow(element.rowIndex);


   var children = t.rows;
   
   var i = 0;
   for (i = 0; i < children.length; i ++) {
      if (children[i].getAttribute("colorname") > name)
         break;
   }

   var row = t.insertRow(i);
   row.id = "colorMap" + name;
   row.setAttribute("colorname", name);
   var cell = row.insertCell(0);
   cell.setAttribute("class", "border");
   cell.innerHTML = "<div style='position:relative'><div style='padding-top:32px; padding-bottom:16px'><img src=" + filename + " width=20 height=100></div><div style='position:absolute; top:0px'>" + name + "</div><div style='position:absolute; top:16px'>" + max + "</div><div style='position:absolute; top:132px'>" + min + "</div> </div>";
}

/* create buffers for the quad interactor */
function createQuad()
{
   quad.buffers = { };
   quad.normals = [ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 ];
   quad.vertices = [ -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0 ];
   quad.indices = [ 0, 1, 3, 1, 2, 3 ];
   quad.colors = [ 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 0.2 ];

   quad.buffers.normals = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, quad.buffers.normals);
   gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(quad.normals), gl.STATIC_DRAW);
         
   quad.buffers.vertices = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, quad.buffers.vertices);
   gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(quad.vertices), gl.STATIC_DRAW);
         
   quad.buffers.colors = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, quad.buffers.colors);
   gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(quad.colors), gl.STATIC_DRAW);

   quad.buffers.indices = gl.createBuffer();
   gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, quad.buffers.indices);
   gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new WebGLUnsignedShortArray(quad.indices), gl.STREAM_DRAW);

   quad.timestep = -1;
   quad.type = "triangles";
}

/* create buffers for the logo */
function createLogo()
{
   logo.buffers = { };
   logo.vertices = [ 0.6, -1.0, -1.0,
                     1.0, -1.0, -1.0,
                     1.0, -0.84, -1.0,
                     0.6, -0.84, -1.0 ];
   logo.indices = [ 0, 1, 3, 1, 2, 3 ];
   logo.texcoord = [ 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 ];
   logo.buffers.vertices = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, logo.buffers.vertices);
   gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(logo.vertices), gl.STATIC_DRAW);

   logo.buffers.indices = gl.createBuffer();
   gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, logo.buffers.indices);
   gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new WebGLUnsignedShortArray(logo.indices), gl.STREAM_DRAW);

   logo.buffers.texcoord = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, logo.buffers.texcoord);
   gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(logo.texcoord), gl.STATIC_DRAW);

   logo.texture = gl.createTexture();
   logo.texture.image = new Image();
   logo.texture.image.onload = function() {
      handleTexture(logo.texture);
      draw();
   }
   logo.texture.image.src = "/logo.png";

   logo.timestep = -1;
   logo.type = "triangles";
}

function handleTexture(texture) {
   gl.bindTexture(gl.TEXTURE_2D, texture);
   gl.texImage2D(gl.TEXTURE_2D, 0, texture.image, true);
   gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
   gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
   gl.bindTexture(gl.TEXTURE_2D, null);
}

/* create buffers for the sphere interactor */
function createSphere()
{
   var latitudeBands = 20;
   var longitudeBands = 20;
   var radius = 0.01;
         
   sphere.buffers = { };
   sphere.normals = [];
   sphere.vertices = [];
   sphere.indices = [];
   sphere.colors = [];

   for (var latNumber = 0; latNumber <= latitudeBands; latNumber++) {
      var theta = latNumber * Math.PI / latitudeBands;
      var sinTheta = Math.sin(theta);
      var cosTheta = Math.cos(theta);
            
      for (var longNumber = 0; longNumber <= longitudeBands; longNumber++) {
         var phi = longNumber * 2 * Math.PI / longitudeBands;
         var sinPhi = Math.sin(phi);
         var cosPhi = Math.cos(phi);
               
         var x = cosPhi * sinTheta;
         var y = cosTheta;
         var z = sinPhi * sinTheta;
         var u = 1- (longNumber / longitudeBands);
         var v = latNumber / latitudeBands;
               
         sphere.normals.push(x);
         sphere.normals.push(y);
         sphere.normals.push(z);

         sphere.vertices.push(radius * x);
         sphere.vertices.push(radius * y);
         sphere.vertices.push(radius * z);

         sphere.colors.push(1.0);
         sphere.colors.push(1.0);
         sphere.colors.push(1.0);
         sphere.colors.push(1.0);
      }
   }
 
   for (var latNumber = 0; latNumber < latitudeBands; latNumber++) {
      for (var longNumber = 0; longNumber < longitudeBands; longNumber++) {
         var first = (latNumber * (longitudeBands + 1)) + longNumber;
         var second = first + longitudeBands + 1;
         sphere.indices.push(first);
         sphere.indices.push(second);
         sphere.indices.push(first + 1);
               
         sphere.indices.push(second);
         sphere.indices.push(second + 1);
         sphere.indices.push(first + 1);
      }
   }
         
   sphere.buffers.normals = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, sphere.buffers.normals);
   gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(sphere.normals), gl.STATIC_DRAW);
         
   sphere.buffers.vertices = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, sphere.buffers.vertices);
   gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(sphere.vertices), gl.STATIC_DRAW);
         
   sphere.buffers.colors = gl.createBuffer();
   gl.bindBuffer(gl.ARRAY_BUFFER, sphere.buffers.colors);
   gl.bufferData(gl.ARRAY_BUFFER, new WebGLFloatArray(sphere.colors), gl.STATIC_DRAW);

   sphere.buffers.indices = gl.createBuffer();
   gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sphere.buffers.indices);
   gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new WebGLUnsignedShortArray(sphere.indices), gl.STREAM_DRAW);

   sphere.timestep = -1;
   sphere.type = "triangles";
   //objects["sphere"] = sphere;
}

/* add an interactor for an object
 *    currently supported: CuttingSurface point argument */
function addInteractor()
{
   var d = "";
   if (arguments[0] == "XCuttingSurface") {
      for (index = 2; index < arguments.length; index ++) {
         var arg = arguments[index];
         if (arg[0] == "point") {
            var arr = arg[2].split(" ");
            var m = Matrix.Translation($V([arr[1], arr[2], arr[3]])).ensure4x4();
            interactors[arguments[1] + "!" + arg[0]] = { }
            interactors[arguments[1] + "!" + arg[0]].matrix = m;
            interactors[arguments[1] + "!" + arg[0]].name = arguments[1] + "!" + arg[0];
         }
      }
      if (typeof objects[arguments[1]].interactors == "undefined")
         objects[arguments[1]].interactors = new Array();
      objects[arguments[1]].interactors.push(arguments[1] + "!point");

      addInteractorCheckbox(arguments[1] + "!" + "point", false);
   } else if (arguments[0] == "XLabel") {
      /*
      for (index = 2; index < arguments.length; index ++) {
         var arg = arguments[index];
         if (arg[0] == "point") {
            var arr = arg[2].split(" ");
            var m = Matrix.Translation($V([arr[1], arr[2], arr[3]])).ensure4x4();
            interactors[arguments[1] + "!" + arg[0]] = { }
            interactors[arguments[1] + "!" + arg[0]].matrix = m;
            interactors[arguments[1] + "!" + arg[0]].name = arguments[1] + "!" + arg[0];
            interactors[arguments[1] + "!" + arg[0]].constantSize = 1;

            objects[arguments[1]].matrix = m;
            objects[arguments[1]].constantSize = 1;
         }
      }
      addInteractorCheckbox(arguments[1] + "!" + "point", false);
      */
   } else if (arguments[0] == "XTracer") {
      for (index = 2; index < arguments.length; index ++) {
         var arg = arguments[index];
         if (arg[0] == "startpoint1" || arg[0] == "startpoint2") {
            var arr = arg[2].split(" ");
            var m = Matrix.Translation($V([arr[1], arr[2], arr[3]])).ensure4x4();
            interactors[arguments[1] + "!" + arg[0]] = { }
            interactors[arguments[1] + "!" + arg[0]].matrix = m;
            interactors[arguments[1] + "!" + arg[0]].name = arguments[1] + "!" + arg[0];
         }
      }
      if (typeof objects[arguments[1]].interactors == "undefined")
         objects[arguments[1]].interactors = new Array();

      objects[arguments[1]].interactors.push(arguments[1] + "!startpoint1");
      objects[arguments[1]].interactors.push(arguments[1] + "!startpoint2");

      addInteractorCheckbox(arguments[1] + "!" + "startpoint1", false);
      addInteractorCheckbox(arguments[1] + "!" + "startpoint2", false);
   }
   
   else {
      for (index = 0; index < arguments.length; index ++) {
         console.log(arguments[index]);
      }
   }
}

function addLabel(name)
{
   if (!interactorRequest && window.XMLHttpRequest)
      interactorRequest = new XMLHttpRequest();
   
   var url = "/interactor?name=" + name;
   interactorRequest.abort();               
   interactorRequest.onreadystatechange = null;
   interactorRequest.open("GET", url, true);
   interactorRequest.send(null);
}

function animate()
{
   /*
     if (timesteps != -1) {
     timestep ++;
     if (timestep < 0)
     timestep = 0;
     if (timestep >= timesteps)
     timestep = 0;
     draw();
     }
     setTimeout("animate()", 20);
   */
}

function initGL()
{
   var canvas = document.getElementById("canvas");
   try {
      gl = canvas.getContext("webkit-3d");
   } catch(e) {}
   if (!gl) {
      try {
         gl = canvas.getContext("moz-webgl");
      } catch(e) {}
   }

   if (!gl)
      alert("Could not initialise WebGL");

   if (!gl.getShaderi)
      gl.getShaderi = gl.getShaderParameter;

   if (!gl.getProgrami)
      gl.getProgrami = gl.getProgramParameter;

   createSphere();
   createQuad();
   createLogo();

   canvas.addEventListener("DOMMouseScroll", function(ev) {

         slider.setRealValue(timestep - ev.detail / 3);
         return true;
      }, false);

   canvas.addEventListener("mousedown", function(ev) {

         mouseDown = true;
         last.setElements([ev.layerX, ev.layerY]);
          
         pick(ev.layerX, ev.layerY);
         draw();

         return true;
      }, false);

   canvas.addEventListener("mousemove", function(ev) {

         if (!mouseDown)
            return false;

         var selected;
         var selection = document.interactor.inter;
         for (i = 1; i < selection.length; i ++)
            {
               if (selection[i].checked)
                  selected = selection[i].value;
            }
          
         rot = view.minor(1, 1, 3, 3).ensure4x4();
         trans = Matrix.Translation($V([view.e(1, 4), view.e(2, 4), view.e(3, 4)])).ensure4x4();

         if (ev.ctrlKey) {
            var m = Matrix.Translation($V([0, 0, (ev.layerX - last.e(1)) / 20])).ensure4x4();
            trans = m.x(trans);
         } else if (ev.shiftKey) {
            var m = Matrix.Translation($V([(ev.layerX - last.e(1)) / 50, -(ev.layerY - last.e(2)) / 50, 0])).ensure4x4();
            trans = m.x(trans);
         } else if (selected == undefined) {
            var m = Matrix.Rotation((ev.layerX - last.e(1)) / 100, $V([0, 1, 0])).ensure4x4();
            m = m.x(Matrix.Rotation((ev.layerY - last.e(2)) / 100, $V([1, 0, 0])).ensure4x4());
            rot = m.x(rot);
         } else {
            var m = $V([(ev.layerX - last.e(1)) / 100, -(ev.layerY - last.e(2)) / 100, 0, 1]);
            m = rot.inverse().x(m);
            interactors[selected].matrix = interactors[selected].matrix.x(Matrix.Translation(m));
         }

         last.setElements([ev.layerX, ev.layerY]);
         view = trans.x(rot);
          
         draw();
         return true;
      }, false);

   canvas.addEventListener("mouseup", function(ev) {
         mouseDown = false;

         var selected;
         var selection = document.interactor.inter;
         for (i = 1; i < selection.length; i ++)
            {
               if (selection[i].checked)
                  selected = selection[i].value;
            }

         if (selected != undefined) {

            var value = interactors[selected].matrix.e(1, 4) + " " + interactors[selected].matrix.e(2, 4) + " " + interactors[selected].matrix.e(3, 4);

            if (!interactorRequest && window.XMLHttpRequest)
               interactorRequest = new XMLHttpRequest();

            var url = "/interactor?name=" + selected + "&value=" + value;
            interactorRequest.abort();               
            interactorRequest.onreadystatechange = null;
            interactorRequest.open("GET", url, true);
            interactorRequest.send(null);
         }
         /*
           if (!viewRequest && window.XMLHttpRequest)
           viewRequest = new XMLHttpRequest();

           var url = "/view?mvm=" + view.elements;
           viewRequest.abort();               
           viewRequest.onreadystatechange = null;
           viewRequest.open("GET", url, true);
           viewRequest.send(null);
         */
      }, false);
}

function loadIdentity()
{
   mvMatrix = Matrix.I(4);
}

/* multiply modelview matrix with given matrix */
function multMatrix(m)
{
   mvMatrix = mvMatrix.x(m);
}

/* translate modelview matrix */
function mvTranslate(v)
{
   var m = Matrix.Translation($V([v[0], v[1], v[2]])).ensure4x4();
   multMatrix(m);
}

/* create perspective matrix */
function perspective(fovy, aspect, znear, zfar)
{
   pMatrix = makePerspective(fovy, aspect, znear, zfar);
}

/* set modelview and perspective matrix parameter in active shader */
function setMatrixUniforms()
{
   gl.uniformMatrix4fv(gl.getUniformLocation(currentShader, "uPMatrix"), false, new WebGLFloatArray(pMatrix.flatten()));
   gl.uniformMatrix4fv(gl.getUniformLocation(currentShader, "uMVMatrix"), false, new WebGLFloatArray(mvMatrix.flatten()));
}

/* rotate modelview matrix 
 *    ang: angle of rotation in degrees
 *    v:   rotation axis
 */
function mvRotate(ang, v)
{
   var arad = ang * Math.PI / 180.0;
   var m = Matrix.Rotation(arad, $V([v[0], v[1], v[2]])).ensure4x4();
   multMatrix(m);
}

/* render COVISE object
 *    obj: the object
 *    pick: enable picking
 */
function drawObject(obj, pick) {

   var lv = Vector.create([0.0, 0.0, -1.0, 0.0]);
   lv = mvMatrix.inverse().multiply(lv);
         
   gl.bindBuffer(gl.ARRAY_BUFFER, obj.buffers.vertices);
   gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
   gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, obj.buffers.indices);
         
   if (pick) {
      useShader(colorPickShader);
      pickIndex ++;
      gl.uniform4f(gl.getUniformLocation(currentShader, "color"), pickIndex / 255.0, 0, 0, 1.0);
            
   } else if (typeof obj.buffers.texcoord == "undefined") {
      if (typeof obj.buffers.normals == "undefined")
         useShader(colorFlatShader);
      else
         useShader(colorShader);
   } else {
      if (typeof obj.buffers.normals == "undefined")
         useShader(textureFlatShader);
      else
         useShader(textureShader);
   }
         
   if (currentShader == colorShader || currentShader == colorFlatShader || currentShader == colorPickShader) {
      if (obj.buffers.colors) {
         gl.bindBuffer(gl.ARRAY_BUFFER, obj.buffers.colors);
         gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
         gl.enableVertexAttribArray(1);
      }
   }
         
   if (currentShader == colorShader) {
      if (obj.buffers.normals) {
         gl.bindBuffer(gl.ARRAY_BUFFER, obj.buffers.normals);
         gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);
         gl.enableVertexAttribArray(2);
      }
   }
         
   if (currentShader == textureShader || currentShader == textureFlatShader) {
      if (obj.buffers.texcoord) {
         gl.bindBuffer(gl.ARRAY_BUFFER, obj.buffers.texcoord);
         gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
         gl.enableVertexAttribArray(1);
               
         gl.activeTexture(gl.TEXTURE0);
         gl.bindTexture(gl.TEXTURE_2D, obj["texture"]);
         gl.uniform1f(gl.getUniformLocation(currentShader, "uSampler"), 0);
      }
   }

   if (currentShader != colorFlatShader && currentShader != textureFlatShader && currentShader != colorPickShader)
      gl.uniform3f(gl.getUniformLocation(currentShader, "uLightDir"), lv.e(1), lv.e(2), lv.e(3));
         
   setMatrixUniforms();
         
   if (obj["type"] == "triangles")
      gl.drawElements(gl.TRIANGLES, obj["indices"].length, gl.UNSIGNED_SHORT, 0);
   else if (obj["type"] == "lines")
      gl.drawElements(gl.LINES, obj["indices"].length, gl.UNSIGNED_SHORT, 0);
}

/* clear framebuffer and render all objects & interactors */
function draw()
{
   gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

   gl.enable(gl.DEPTH_TEST);
   gl.depthFunc(gl.LEQUAL);

   gl.enableVertexAttribArray(0);

   perspective(45, 1.0, 0.1, 100.0);
   loadIdentity();

   multMatrix(view);

   for (var object in objects)
      if (objects[object] && (objects[object]["timestep"] == -1 || objects[object]["timestep"] == timestep) && typeof objects[object]["alpha"] == "undefined")
         drawObject(objects[object], false);

   for (var object in interactors) {
      multMatrix(interactors[object].matrix);
      if (interactors[object].constantSize) {
         var scale = - trans.e(3, 4) / 8;
         multMatrix(Matrix.Scale($V([scale, scale, scale])));
      }
      drawObject(sphere, false);
      loadIdentity();
      multMatrix(view);
   }

   gl.enable(gl.BLEND);
   gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

   for (var object in objects)
      if (objects[object] && (objects[object]["timestep"] == -1 || objects[object]["timestep"] == timestep) && typeof objects[object]["alpha"]) {

         // labels
         if (objects[object].matrix) {
            multMatrix(objects[object].matrix);
            multMatrix(rot.inverse());
            if (objects[object].constantSize) {
               var scale = - trans.e(3, 4) / 20;
               multMatrix(Matrix.Scale($V([scale, scale, scale])));
               multMatrix(Matrix.Translation($V([0.0, 1.2, 0.0])));
            }
         }
         drawObject(objects[object], false);
         loadIdentity();
         multMatrix(view);
      }

   gl.disable(gl.BLEND);

   for (var object in interactors) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      multMatrix(interactors[object].matrix);
      loadIdentity();
      multMatrix(view);
   }

   pMatrix = Matrix.I(4);
   loadIdentity();
   gl.enable(gl.BLEND);
   drawObject(logo, false);
   gl.disable(gl.BLEND);
}

/* pick at given location
 *    if an interactor is picked, its checkbox is selected.
 *    x, y: mouse position
 */
function pick(x, y)
{
   gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
   gl.scissor(x, 600 - y, 1, 1);
   gl.enable(gl.SCISSOR_TEST);
   
   perspective(45, 1.0, 0.1, 100.0);
   loadIdentity();

   multMatrix(view);

   gl.enable(gl.DEPTH_TEST);
   gl.depthFunc(gl.LEQUAL);
   gl.disable(gl.BLEND);

   gl.enableVertexAttribArray(0);

   pickIndex = 0;
   for (var object in interactors) {
      multMatrix(interactors[object].matrix);
      drawObject(sphere, true);
      loadIdentity();
      multMatrix(view);
   }
   var data = gl.readPixels(x, 600 - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE);
   if (data.data)
      data = data.data;

   if (data[0] > 0) {
      var index = 1;
      for (var object in interactors) {
         if (index == data[0])
            selectInteractorCheckbox(interactors[object].name);
         index ++;
      }
   }

   gl.disable(gl.SCISSOR_TEST);
}

/* select an interactor's checkbox */
function selectInteractorCheckbox(name)
{
   var selected;
   var selection = document.interactor.inter;
   for (i = 1; i < selection.length; i ++)
      {
         if (selection[i].value == name)
            selection[i].checked = true;
      }
}

/* delete an interactor's checkbox */
function deleteInteractorCheckbox(name)
{
   var element = document.getElementById(name);
   var parent = document.getElementById("interactor");
   if (parent && element)         
      parent.removeChild(element);
}

/* add an interactor's checkbox */
function addInteractorCheckbox(name, checked)
{
   var form = document.getElementById("interactor");
   var children = form.elements;
   /*
   var i = 0;
   for (i = 1; i < children.length; i ++) {
      //console.log("  " + children[i].getAttribute("value"));
      if (children[i].getAttribute("value") > name)
         break;
   }
   */
   var radioHtml = '<input type="radio" name="inter" value="' + name + '"';
   if (checked)
      radioHtml += 'checked="checked"/>' + name;
   else
      radioHtml += '>' + name;

   var radioFragment = document.createElement('div');
   radioFragment.id = "checkbox" + name;
   radioFragment.innerHTML = radioHtml;
   /*
   if (children.length > 0)
      form.insertBefore(radioFragment, children[i]);
   else
   */
   form.appendChild(radioFragment);
}

function webGLStart()
{
   initGL();
   initShaders();

   gl.clearColor(0.0, 0.0, 0.0, 1.0);

   if (gl.clearDepthf)
      gl.clearDepthf(1.0);
   else
      gl.clearDepth(1.0);

   gl.enable(gl.DEPTH_TEST);
   gl.depthFunc(gl.LEQUAL);

   draw();
}

function init()
{
   addInteractorCheckbox("No Interactor", true);
   webGLStart();

   //getData();
   webSocket();
}
