/*
move an editable Spline to a polygon surface below the editable spline
*/
macroscript MoveToSurface category: "Terrain"
(
fn g_filter o = superclassof o == Geometryclass
fn find_intersection targetNode origPos =
(
 local testRay = ray origPos [0,0,100000.0]
 intersectRay targetNode testRay
)
on isEnabled return
(
selection.count == 1 and  (classof selection[1] == SplineShape or classof selection[1] == Line)
)
on Execute do
(
 target_mesh = pickObject message:"Pick Target Surface:" filter:g_filter
	
 local face_selection = #{}
 local base_obj = $.baseobject
 if isValidNode target_mesh then
 (
  undo "MoveToSurface" on
  (
   new_z = $.pos.z
 local num_Splines = numSplines $
  for s = 1 to num_Splines do
  (
 local num_Knots = numKnots $ s
	  local lowestZ = 1000.0
    for k = 1 to num_Knots do
    (
      knt = getKnotPoint $ s k
		local testRay = ray knt [0,0,-100000.0]
      newHeight = intersectRay target_mesh  testRay
		if newHeight != undefined and newHeight.pos.z < lowestZ then
		(
		 lowestZ = newHeight.pos.z ;
		)
    )--end k loop
    for k = 1 to num_Knots do
    (
      knt = getKnotPoint $ s k
      in_vec = getInVec $ s k
      out_vec = getOutVec $ s k
      knt.z = in_vec.z = out_vec.z = lowestZ
      setInVec $ s k in_vec
      setOutVec $ s k out_vec
      setKnotPoint $ s k knt
    )--end k loop
  )--end s loop
  updateshape $
  )--end undo
 )--end if
)--end execute
)--end script

