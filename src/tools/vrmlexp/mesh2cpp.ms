-----------------------------------------------------------
-- mesh2cpp.ms
-- By Simon Feltman
-----------------------------------------------------------
macroScript Mesh2CPP
   category:"SDK Tools"
   tooltip:"Mesh to CPP"
(
   fn ExportCPP obj stream =
   (
      format "\tmesh.setNumVerts(%);\n" \
         obj.numverts to:stream

      format "\tmesh.setNumFaces(%);\n" \
         obj.numfaces to:stream

      for i = 1 to obj.numverts do
      (
         pnt = getvert obj i
         format "\tmesh.setVert(%,size*Point3(%,%,%));\n" \
            (i-1) pnt.x pnt.y pnt.z to:stream
      )

      for i = 1 to obj.numfaces do
      (
         pnt = getface obj i
         format "\tmesh.faces[%].setVerts(%, %, %);\n" \
            (i-1) \
            (pnt.x as integer - 1) \
            (pnt.y as integer - 1) \
            (pnt.z as integer - 1) \
            to:stream

         format "\tmesh.faces[%].setEdgeVisFlags(%,%,%);\n"\
            (i-1) \
            (if (getedgevis obj i 1) then 1 else 0) \
            (if (getedgevis obj i 2) then 1 else 0) \
            (if (getedgevis obj i 3) then 1 else 0) \
            to:stream

         format "\tmesh.faces[%].setSmGroup(%);\n" \
            (i-1) (getfacesmoothgroup obj i) to:stream
      )
   )

   if selection[1] != undefined and
      (classof selection[1]) == Editable_mesh then
   (
      local fname = \
         getsavefilename types:"Header Files (*.h)|*.h|"
      local stream = createfile fname
      if stream != undefined then
      (
         ExportCPP (selection[1]) stream
         close stream
      )
      else
         messagebox ("Error opening file \"" + fname + "\"")
   )
   else
      messagebox "You must select an editable mesh."
)
