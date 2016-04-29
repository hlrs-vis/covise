C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE GRAPH_ENDE(lnods,lnods_num,coord_num,
     *                      ifine_haupt,iende_haupt,
     *                      iknmat,iknmat_adr,
     *                      redu_graph,grad_max,grad_min,
     *                      isp,lmax,speich_max_sub)
  
      implicit none     

      include 'mpif.h'
      include 'common.zer'

      integer lnods,coord_num,lnods_num

      integer ifine_haupt,iende_haupt,
     *        isp,lmax,speich_max_sub,
     *        iknmat,iknmat_adr,nl_knmat,
     *        iende,ifine,speich_max

      integer grad_max,grad_min

      integer iint1_help,iint2_help,
     *        ikelem,ikelem_adr,nl_kelem

      logical schreiben,redu_graph
 
      dimension lnods(nelem_max,nkd),coord_num(npoin),lnods_num(nelem)

      dimension isp(lmax)
c     *****************************************************************


      iende=1
      ifine=lmax
      speich_max=0


c     ****************************************************************
c     BERECHNUNG DER MATRIX-STRUKTUR:

      if (myid.eq.0) write(6,*) '                               '
      if (myid.eq.0) write(6,*) 'BERECHUNG DER MATRIX-STRUKTUR '


      CALL ALLOC_FINE(ifine,iint1_help,npoin+1)
      CALL ALLOC_FINE(ifine,iint2_help,npoin+1)

c     Bestimmung der an den Knoten beteiligten Elemente:
      nl_kelem=nelem*nkd
      CALL ALLOC_FINE(ifine,ikelem,nl_kelem)
      CALL ALLOC_FINE(ifine,ikelem_adr,npoin+1)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

      CALL KELE(lnods,lnods_num,coord_num,
     *          nelem,nelem_max,nkd,
     *          isp(ikelem),isp(ikelem_adr),nl_kelem,
     *          isp(iint1_help),npoin,myid,parallel)

c     Bestimmung der Matrix-Struktur ohne Kopplungen:    
      CALL ALLOC_FINE(ifine,iknmat,0)
      CALL ALLOC_FINE(ifine,iknmat_adr,0)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

      schreiben=.false.
      CALL GRAPH_KAN(lnods,coord_num,nelem,nelem_max,nkd,
     *               isp(ikelem),isp(ikelem_adr),nl_kelem,
     *               isp(iknmat),isp(iknmat_adr),nl_knmat,
     *               isp(iint1_help),isp(iint2_help),npoin,
     *               lnods_num,grad_max,grad_min,
     *               myid,parallel,lupar,redu_graph,schreiben)

      CALL ALLOC_ENDE(iende,iknmat,nl_knmat)
      CALL ALLOC_ENDE(iende,iknmat_adr,npoin+1)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

      schreiben=.true. 
      CALL GRAPH_KAN(lnods,coord_num,nelem,nelem_max,nkd,
     *               isp(ikelem),isp(ikelem_adr),nl_kelem,
     *               isp(iknmat),isp(iknmat_adr),nl_knmat,
     *               isp(iint1_help),isp(iint2_help),npoin,
     *               lnods_num,grad_max,grad_min,
     *               myid,parallel,lupar,redu_graph,schreiben)

      if (myid.eq.0) write(6,*) 'BERECHUNG DER MATRIX-STRUKTUR BEENDET'

      CALL DEALLOC_ALLE(ifine,lmax)
c     ****************************************************************


c     *****************************************************************
c     BESTIMMUNG DER ADRESSEN DER FELDER IM HAUPTPROGRAMM:

      nl_kompakt=nl_knmat

      iknmat    =iknmat+iende_haupt-1
      iknmat_adr=iknmat_adr+iende_haupt-1

      iende_haupt=iende+iende_haupt-1
c     *****************************************************************


      speich_max_sub=speich_max
      return
      end

