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

#include "stdlib.h"
#include "minimize.h"
#include "domain.h"
#include "update.h"
#include "min.h"
#include "finish.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

Minimize::Minimize(LAMMPS *lmp) : Pointers(lmp) {}

/* ---------------------------------------------------------------------- */

void Minimize::command(int narg, char **arg)
{
  if (narg != 4) error->all("Illegal minimize command");

  if (domain->box_exist == 0)
    error->all("Minimize command before simulation box is defined");

  update->etol = atof(arg[0]);
  update->ftol = atof(arg[1]);
  update->nsteps = atoi(arg[2]);
  update->max_eval = atoi(arg[3]);

  if (update->etol < 0.0 || update->ftol < 0.0)
    error->all("Illegal minimize command");

  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + update->nsteps;

  update->whichflag = 1;

  lmp->init();
  update->minimize->run();

  Finish finish(lmp);
  finish.end(1);

  update->whichflag = -1;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}
