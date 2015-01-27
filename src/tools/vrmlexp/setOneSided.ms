macroScript setOneSided category:"Material" 
(
        for x in sceneMaterials do
		(
			  
					print x.twoSided
			x.twoSided = true
		)
)