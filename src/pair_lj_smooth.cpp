/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Craig Maloney (UCSB)
------------------------------------------------------------------------- */
#include "mpi.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "pair_lj_smooth.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* ---------------------------------------------------------------------- */

PairLJSmooth::PairLJSmooth(LAMMPS *lmp) : Pair(lmp) {}

/* ---------------------------------------------------------------------- */

PairLJSmooth::~PairLJSmooth()
{
  if (allocated) {
    memory->destroy_2d_int_array(setflag);
    memory->destroy_2d_double_array(cutsq);

    memory->destroy_2d_double_array(cut);
    memory->destroy_2d_double_array(cut_inner);
    memory->destroy_2d_double_array(cut_inner_sq);
    memory->destroy_2d_double_array(epsilon);
    memory->destroy_2d_double_array(sigma);
    memory->destroy_2d_double_array(lj1);
    memory->destroy_2d_double_array(lj2);
    memory->destroy_2d_double_array(lj3);
    memory->destroy_2d_double_array(lj4);
    memory->destroy_2d_double_array(ljsw0);
    memory->destroy_2d_double_array(ljsw1);
    memory->destroy_2d_double_array(ljsw2);
    memory->destroy_2d_double_array(ljsw3);
    memory->destroy_2d_double_array(ljsw4);
    memory->destroy_2d_double_array(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJSmooth::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj;
  double r,t,tsq,fskin;
  int *ilist,*jlist,*numneigh,**firstneigh;
  
  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      if (j < nall) factor_lj = 1.0;
      else {
	factor_lj = special_lj[j/nall];
	j %= nall;
      }
      
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      
      if (rsq < cutsq[itype][jtype]) {
	r2inv = 1.0/rsq;
	if (rsq < cut_inner_sq[itype][jtype]) {
	  r6inv = r2inv*r2inv*r2inv;
	  forcelj = r6inv * (lj1[itype][jtype]*r6inv-lj2[itype][jtype]);
	} else {
	  r = sqrt(rsq); 
	  t = r - cut_inner[itype][jtype];
	  tsq = t*t;
	  fskin = ljsw1[itype][jtype] + ljsw2[itype][jtype]*t +
	    ljsw3[itype][jtype]*tsq + ljsw4[itype][jtype]*tsq*t; 
	  forcelj = fskin*r;
	}
        
	fpair = factor_lj*forcelj*r2inv;
        
	f[i][0] += delx*fpair;
	f[i][1] += dely*fpair;
	f[i][2] += delz*fpair;
	if (newton_pair || j < nlocal) {
	  f[j][0] -= delx*fpair;
	  f[j][1] -= dely*fpair;
	  f[j][2] -= delz*fpair;
	}
	
	if (eflag) {
	  if (rsq < cut_inner_sq[itype][jtype])
	    evdwl = r6inv * (lj3[itype][jtype]*r6inv - 
			     lj4[itype][jtype]) - offset[itype][jtype];
	  else
	    evdwl = ljsw0[itype][jtype] - ljsw1[itype][jtype]*t -
	      ljsw2[itype][jtype]*tsq/2.0 - ljsw3[itype][jtype]*tsq*t/3.0 -
	      ljsw4[itype][jtype]*tsq*tsq/4.0 - offset[itype][jtype];
	  evdwl *= factor_lj;
	}

	if (evflag) ev_tally(i,j,nlocal,newton_pair,
			     evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays 
------------------------------------------------------------------------- */

void PairLJSmooth::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  setflag = memory->create_2d_int_array(n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  cutsq = memory->create_2d_double_array(n+1,n+1,"pair:cutsq");

  cut = memory->create_2d_double_array(n+1,n+1,"pair:cut");
  cut_inner = memory->create_2d_double_array(n+1,n+1,"pair:cut_inner");
  cut_inner_sq = memory->create_2d_double_array(n+1,n+1,"pair:cut_inner_sq");
  epsilon = memory->create_2d_double_array(n+1,n+1,"pair:epsilon");
  sigma = memory->create_2d_double_array(n+1,n+1,"pair:sigma");
  lj1 = memory->create_2d_double_array(n+1,n+1,"pair:lj1");
  lj2 = memory->create_2d_double_array(n+1,n+1,"pair:lj2");
  lj3 = memory->create_2d_double_array(n+1,n+1,"pair:lj3");
  lj4 = memory->create_2d_double_array(n+1,n+1,"pair:lj4");
  ljsw0 = memory->create_2d_double_array(n+1,n+1,"pair:ljsw0");
  ljsw1 = memory->create_2d_double_array(n+1,n+1,"pair:ljsw1");
  ljsw2 = memory->create_2d_double_array(n+1,n+1,"pair:ljsw2");
  ljsw3 = memory->create_2d_double_array(n+1,n+1,"pair:ljsw3");
  ljsw4 = memory->create_2d_double_array(n+1,n+1,"pair:ljsw4");
  offset = memory->create_2d_double_array(n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings 
------------------------------------------------------------------------- */

void PairLJSmooth::settings(int narg, char **arg)
{
  if (narg != 2) error->all("Illegal pair_style command");

  cut_inner_global = atof(arg[0]);
  cut_global = atof(arg[1]);

  if (cut_inner_global <= 0.0 || cut_inner_global > cut_global)
    error->all("Illegal pair_style command");

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
	if (setflag[i][j]) {
	  cut_inner[i][j] = cut_inner_global;
	  cut[i][j] = cut_global;
	}
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJSmooth::coeff(int narg, char **arg)
{
  if (narg != 4 && narg != 6)
    error->all("Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = atof(arg[2]);
  double sigma_one = atof(arg[3]);
  
  double cut_inner_one = cut_inner_global;
  double cut_one = cut_global;
  if (narg == 6) {
    cut_inner_one = atof(arg[4]);
    cut_one = atof(arg[5]);
  }

  if (cut_inner_one <= 0.0 || cut_inner_one > cut_one)
    error->all("Incorrect args for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut_inner[i][j] = cut_inner_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all("Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJSmooth::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
			       sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut_inner[i][j] = mix_distance(cut_inner[i][i],cut_inner[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  cut_inner_sq[i][j] = cut_inner[i][j]*cut_inner[i][j];
  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (cut_inner[i][j] != cut[i][j]) {
    double r6inv = 1.0/pow(cut_inner[i][j],6.0);
    double t = cut[i][j] - cut_inner[i][j];
    double tsq = t*t;
    double ratio = sigma[i][j] / cut_inner[i][j];
    ljsw0[i][j] = 4.0*epsilon[i][j]*(pow(ratio,12.0) - pow(ratio,6.0));
    ljsw1[i][j] = r6inv*(lj1[i][j]*r6inv-lj2[i][j]) / cut_inner[i][j];
    ljsw2[i][j] = -r6inv * (13.0*lj1[i][j]*r6inv - 7.0*lj2[i][j]) /
      cut_inner_sq[i][j];
    ljsw3[i][j] = -(3.0/tsq) * (ljsw1[i][j] + 2.0/3.0*ljsw2[i][j]*t);
    ljsw4[i][j] = -1.0/(3.0*tsq) * (ljsw2[i][j] + 2.0*ljsw3[i][j]*t);
    if (offset_flag)
      offset[i][j] = ljsw0[i][j] - ljsw1[i][j]*t - ljsw2[i][j]*tsq/2.0 -
	ljsw3[i][j]*tsq*t/3.0 - ljsw4[i][j]*tsq*tsq/4.0;
    else offset[i][j] = 0.0;
  } else {
    ljsw0[i][j] = 0.0;
    ljsw1[i][j] = 0.0;
    ljsw2[i][j] = 0.0;
    ljsw3[i][j] = 0.0;
    ljsw4[i][j] = 0.0;
    double ratio = sigma[i][j] / cut_inner[i][j];
    if (offset_flag)
      offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
    else offset[i][j] = 0.0;
  }
              
  cut_inner[j][i] = cut_inner[i][j];
  cut_inner_sq[j][i] = cut_inner_sq[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  ljsw0[j][i] = ljsw0[i][j];
  ljsw1[j][i] = ljsw1[i][j];
  ljsw2[j][i] = ljsw2[i][j];
  ljsw3[j][i] = ljsw3[i][j];
  ljsw4[j][i] = ljsw4[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file 
------------------------------------------------------------------------- */

void PairLJSmooth::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
	fwrite(&epsilon[i][j],sizeof(double),1,fp);
	fwrite(&sigma[i][j],sizeof(double),1,fp);
	fwrite(&cut_inner[i][j],sizeof(double),1,fp);
	fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJSmooth::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
	if (me == 0) {
	  fread(&epsilon[i][j],sizeof(double),1,fp);
	  fread(&sigma[i][j],sizeof(double),1,fp);
	  fread(&cut_inner[i][j],sizeof(double),1,fp);
	  fread(&cut[i][j],sizeof(double),1,fp);
	}
	MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&cut_inner[i][j],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJSmooth::write_restart_settings(FILE *fp)
{
  fwrite(&cut_inner_global,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJSmooth::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_inner_global,sizeof(double),1,fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_inner_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairLJSmooth::single(int i, int j, int itype, int jtype, double rsq,
			    double factor_coul, double factor_lj,
			    double &fforce)
{
  double r2inv,r6inv,forcelj,philj,r,t,tsq,fskin;

  r2inv = 1.0/rsq;
  if (rsq < cut_inner_sq[itype][jtype]) {
    r6inv = r2inv*r2inv*r2inv;
    forcelj = r6inv * (lj1[itype][jtype]*r6inv-lj2[itype][jtype]);
  } else {
    r = sqrt(rsq); 
    t = r - cut_inner[itype][jtype];
    tsq = t*t;
    fskin = ljsw1[itype][jtype] + ljsw2[itype][jtype]*t + 
      ljsw3[itype][jtype]*tsq + ljsw4[itype][jtype]*tsq*t; 
    forcelj = fskin*r;
  }
  fforce = factor_lj*forcelj*r2inv;
    
  if (rsq < cut_inner_sq[itype][jtype])
    philj = r6inv * (lj3[itype][jtype]*r6inv - lj4[itype][jtype]) -
      offset[itype][jtype];
  else
    philj = ljsw0[itype][jtype] - ljsw1[itype][jtype]*t -
      ljsw2[itype][jtype]*tsq/2.0 - ljsw3[itype][jtype]*tsq*t/3.0 -
      ljsw4[itype][jtype]*tsq*tsq/4.0 - offset[itype][jtype];
  return factor_lj*philj;
}
