C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE KN_CHECK(kn_num,iadr,coord_num,coord_zeig,coord,
     *                                                     vec)

      implicit none
      include 'common.zer'

      integer  kn_num,iadr,coord_num,coord_zeig,
     *         luerr,j

      real    coord,vec

      dimension coord(npoin_max,ncd),coord_num(npoin_max),
     *          coord_zeig(knmax_num)
      dimension vec(3)
c     ****************************************************************

c     ****************************************************************

      if (kn_num.gt.knmax_num) then
             call erro_init(myid,parallel,luerr)
             write(luerr,*)'FEHLER IN SUBROUTINE KN_CHECK          '
             write(luerr,*)'                            '
             write(luerr,*)'Eine eingelesene Knotennummer ist groesser'
             write(luerr,*)'als die im Geometriefile angegebene '
             write(luerr,*)'maximale Knotennummer.              '
             write(luerr,*)'Eingelesene Knotennummer     :',kn_num    
             write(luerr,*)'Angegebene max.  Knotennummer:',knmax_num 
             write(luerr,*)'Adresse                      :',iadr
             call erro_ende(myid,parallel,luerr)
      endif

      IF (coord_zeig(kn_num).ne.0) THEN
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'FEHLER IM GEOMETRIE-FILE !! '
         write(luerr,*)'                            '
         write(luerr,*)'Die Knotennumer ',kn_num,' ist doppelt belegt !'
         write(luerr,*)'                                          '
         write(luerr,*)'1.Koordinaten:                            '
         write(luerr,999) (coord(coord_zeig(kn_num),j),j=1,ncd)
         write(luerr,*)'2.Koordinaten:                            '
         write(luerr,999) (vec(j),j=1,ncd)
         call erro_ende(myid,parallel,luerr)
      ENDIF
c     ****************************************************************

 999  format(3(e10.4,1x))

      return
      end

