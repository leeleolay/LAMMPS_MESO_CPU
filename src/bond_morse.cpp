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
   Contributing author: Jeff Greathouse (SNL)
------------------------------------------------------------------------- */
#include "mpi.h"
#include "math.h"
#include "stdlib.h"
#include "bond_morse.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondMorse::BondMorse(LAMMPS *lmp) : Bond(lmp) {}

/* ---------------------------------------------------------------------- */

BondMorse::~BondMorse()
{
  if (allocated) {
    memory->sfree(setflag);
    memory->sfree(d0);
    memory->sfree(alpha);
    memory->sfree(r0);
  }
}

/* ---------------------------------------------------------------------- */

void BondMorse::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq,r,dr,ralpha;

  ebond = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    domain->minimum_image(delx,dely,delz);

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    dr = r - r0[type];
    ralpha = exp(-alpha[type]*dr);

    // force & energy

    if (r > 0.0) fbond = -2.0*d0[type]*alpha[type]*(1-ralpha)*ralpha/r;
    else fbond = 0.0;

    if (eflag) ebond = d0[type]*(1-ralpha)*(1-ralpha);

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond;
      f[i1][1] += dely*fbond;
      f[i1][2] += delz*fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond;
      f[i2][1] -= dely*fbond;
      f[i2][2] -= delz*fbond;
    }

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondMorse::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  d0 = (double *) memory->smalloc((n+1)*sizeof(double),"bond:d0");
  alpha = (double *) memory->smalloc((n+1)*sizeof(double),"bond:alpha");
  r0 = (double *) memory->smalloc((n+1)*sizeof(double),"bond:r0");
  setflag = (int *) memory->smalloc((n+1)*sizeof(int),"bond:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void BondMorse::coeff(int narg, char **arg)
{
  if (narg != 4) error->all("Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(arg[0],atom->nbondtypes,ilo,ihi);

  double d0_one = atof(arg[1]);
  double alpha_one = atof(arg[2]);
  double r0_one = atof(arg[3]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    d0[i] = d0_one;
    alpha[i] = alpha_one;
    r0[i] = r0_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all("Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length 
------------------------------------------------------------------------- */

double BondMorse::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void BondMorse::write_restart(FILE *fp)
{
  fwrite(&d0[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&alpha[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&r0[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void BondMorse::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&d0[1],sizeof(double),atom->nbondtypes,fp);
    fread(&alpha[1],sizeof(double),atom->nbondtypes,fp);
    fread(&r0[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&d0[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&alpha[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&r0[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ---------------------------------------------------------------------- */

double BondMorse::single(int type, double rsq, int i, int j)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  double ralpha = exp(-alpha[type]*dr);
  return d0[type]*(1-ralpha)*(1-ralpha);
}
