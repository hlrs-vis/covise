C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE EL_CHECK(el_num,help2,coord_zeig,lnods_zeig,
     *                    coord_num,lnods)

      implicit none

      include 'common.zer'

      integer el_num,help2(8),help1(8),
     *        coord_zeig,lnods_zeig,
     *        lnods,coord_num,
     *        luerr,j,k,ij,el_adr,kn_num


      dimension coord_zeig(knmax_num),lnods_zeig(elmax_num),
     *          coord_num(npoin_max),lnods(nelem_max,nkd)
c     ****************************************************************


c     ****************************************************************
c     KONTROLLE DER ELEMENT-NUMMER:

      if (el_num.gt.elmax_num) then
             call erro_init(myid,parallel,luerr)
             write(luerr,*)'FEHLER IN SUBROUTINE EL_CHECK'
             write(luerr,*)'                            '
             write(luerr,*)'Eine eingelesene Elementnummer ist groesser'
             write(luerr,*)'als die im Geometriefile angegebene '
             write(luerr,*)'maximale Elementnummer.              '
             write(luerr,*)'Eingelesene Elementnummer     :',el_num    
             write(luerr,*)'Angegebene max.  Elementnummer:',elmax_num 
             call erro_ende(myid,parallel,luerr)
      endif


      IF (lnods_zeig(el_num).ne.0) THEN
c       Die Element-Nummer ist bereits belegt. Die Knoten dieses
c       Elementes stehen auf Feld lnods unter der Adress el_adr:
        el_adr=lnods_zeig(el_num)
        do 200 j=1,nkd
          help1(j)=coord_num(lnods(el_adr,j))
 200    continue

        call erro_init(myid,parallel,luerr)
        write(luerr,*)'FEHLER BEI ELEMENT-NUMERIERUNG '
        write(luerr,*)'Die Elementnumer ',el_num,' ist doppelt belegt !'
        write(luerr,*)'Knoten der ersten Belegung:'              
        write(luerr,999) (help1(j),j=1,nkd)
        write(luerr,*)'Knoten der zweiten Belegung:'            
        write(luerr,999) (help2(j),j=1,nkd)
        call erro_ende(myid,parallel,luerr)
      ENDIF
 999  format(8(i7,1x))
c     ****************************************************************


c     ****************************************************************
c     KONTROLLE DER KNOTEN-NUMMERN:

      do 100 j=1,nkd       
         kn_num=help2(j)
         if (kn_num.gt.knmax_num .or. coord_zeig(kn_num).eq.0) THEN
             call erro_init(myid,parallel,luerr)
             write(luerr,*)'FEHLER IN ELEMENT-KNOTEN-LISTE         '
             write(luerr,*)'Die angegebene Knotennummer ', kn_num
             write(luerr,*)'existiert nicht im Geometrie-File bzw.'
             write(luerr,*)'besitzt keine Koordinaten.            '
             call erro_ende(myid,parallel,luerr)
         endif
 100  continue
c     ****************************************************************

c     ****************************************************************
c     KONTROLLE OB KNOTEN MEHRMALS IM ELEMENT VORKOMMEN:
      do 300 j=1,nkd
        do 310 k=1,nkd
          if (k.ne.j) then
             if (help2(j).eq.help2(k)) then
                call erro_init(myid,parallel,luerr)
                write(luerr,*)'FEHLER IN ELEMENT-KNOTEN-LISTE '
                write(luerr,*)'Es gibt identische Knoten.           ' 
                write(luerr,*)'Knoten von Element ',el_num,' :'         
                write(luerr,888) (help2(ij),ij=1,nkd)
                call erro_ende(myid,parallel,luerr)
             endif
          endif
 310    continue 
 300  continue
 888  format(8(i7,1x))
c     ****************************************************************


      return
      end

