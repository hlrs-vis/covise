-------------------------
-- adeskBitmap2Bitmap
-- vivfx.se 2018
-------------------------

(
	mats2convert = #()
	
	for m in sceneMaterials do 
		(
				AdskBitmaps = getClassInstances Autodesk_Map
			--	for Bmap in AdskBitmaps collect mats2convert--do (Replaceinstances Bmap (convertmap Bmap m.name))
				mats2convert = for o in AdskBitmaps where  hasProperty o "Parameters_Source"   collect o
		)

	print mats2convert.count 

	for f = 1 to mats2convert.count do
	(
		inMap = mats2convert[f]
		
		outMap = bitmaptexture()
		outMap.name = inmap.name
		try (outmap.bitmap = inmap.Parameters_Source)
		catch ("No path for [" + inMap.name as string + "] in Material [" + mats2convert[f] + ". Please convert manually.")
		Replaceinstances mats2convert[f] outMap
	)
	
)