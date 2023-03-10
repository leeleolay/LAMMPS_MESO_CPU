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

#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "fix_ave_force.h"
#include "atom.h"
#include "update.h"
#include "respa.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixAveForce::FixAveForce(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 6) error->all("Illegal fix aveforce command");

  vector_flag = 1;
  size_vector = 3;
  scalar_vector_freq = 1;
  extvector = 1;

  xflag = yflag = zflag = 1;
  if (strcmp(arg[3],"NULL") == 0) xflag = 0;
  else xvalue = atof(arg[3]);
  if (strcmp(arg[4],"NULL") == 0) yflag = 0;
  else yvalue = atof(arg[4]);
  if (strcmp(arg[5],"NULL") == 0) zflag = 0;
  else zvalue = atof(arg[5]);

  foriginal_all[0] = foriginal_all[1] = foriginal_all[2] = 0.0;
}

/* ---------------------------------------------------------------------- */

int FixAveForce::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAveForce::init()
{
  if (strcmp(update->integrate_style,"respa") == 0)
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // ncount = total # of atoms in group

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int count = 0;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) count++;
  MPI_Allreduce(&count,&ncount,1,MPI_INT,MPI_SUM,world);
}

/* ---------------------------------------------------------------------- */

void FixAveForce::setup(int vflag)
{
  if (strcmp(update->integrate_style,"verlet") == 0)
    post_force(vflag);
  else
    for (int ilevel = 0; ilevel < nlevels_respa; ilevel++) {
      ((Respa *) update->integrate)->copy_flevel_f(ilevel);
      post_force_respa(vflag,ilevel,0);
      ((Respa *) update->integrate)->copy_f_flevel(ilevel);
    }
}

/* ---------------------------------------------------------------------- */

void FixAveForce::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAveForce::post_force(int vflag)
{
  // sum forces on participating atoms

  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double foriginal[3];
  foriginal[0] = foriginal[1] = foriginal[2] = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      foriginal[0] += f[i][0];
      foriginal[1] += f[i][1];
      foriginal[2] += f[i][2];
    }

  // average the force on participating atoms
  // add in requested amount

  MPI_Allreduce(foriginal,foriginal_all,3,MPI_DOUBLE,MPI_SUM,world);
  double fave[3];
  fave[0] = foriginal_all[0]/ncount + xvalue;
  fave[1] = foriginal_all[1]/ncount + yvalue;
  fave[2] = foriginal_all[2]/ncount + zvalue;

  // set force of all participating atoms to same value
  // only for active dimensions

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (xflag) f[i][0] = fave[0];
      if (yflag) f[i][1] = fave[1];
      if (zflag) f[i][2] = fave[2];
    }
}

/* ---------------------------------------------------------------------- */

void FixAveForce::post_force_respa(int vflag, int ilevel, int iloop)
{
  // ave + extra force on outermost level
  // just ave on inner levels

  if (ilevel == nlevels_respa-1) post_force(vflag);
  else {
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    double foriginal[3];
    foriginal[0] = foriginal[1] = foriginal[2] = 0.0;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	foriginal[0] += f[i][0];
	foriginal[1] += f[i][1];
	foriginal[2] += f[i][2];
      }

    MPI_Allreduce(foriginal,foriginal_all,3,MPI_DOUBLE,MPI_SUM,world);
    double fave[3];
    fave[0] = foriginal_all[0]/ncount;
    fave[1] = foriginal_all[1]/ncount;
    fave[2] = foriginal_all[2]/ncount;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
	if (xflag) f[i][0] = fave[0];
	if (yflag) f[i][1] = fave[1];
	if (zflag) f[i][2] = fave[2];
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixAveForce::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixAveForce::compute_vector(int n)
{
  return foriginal_all[n];
}
