
from ortools.sat.python import cp_model
import logging

logger = logging.getLogger(__name__)

class LogisticsSolver:
    """
    Neuro-Symbolic Logistics Solver using Google OR-Tools.
    Translates high-level constraints into exact mathematical schedules.
    """

    def solve_schedule(self, jobs, crews, time_slots):
        """
        Solve a scheduling problem.
        
        Args:
            jobs: List of dicts {id, duration, priority, skills_required, deadline}
            crews: List of dicts {id, skills, availability_start, availability_end}
            time_slots: List of time slots (e.g., hours or days)
            
        Returns:
            Optimal schedule or None
        """
        model = cp_model.CpModel()
        
        # Variables
        # schedule[(job_id, crew_id, start_time)] = boolean
        schedule = {}
        
        # Constraints
        # 1. Each job must be assigned exactly once
        for job in jobs:
            assignments = []
            for crew in crews:
                # Check skills match
                if not set(job.get('skills_required', [])).issubset(set(crew.get('skills', []))):
                    continue
                    
                for t in range(len(time_slots) - job['duration'] + 1):
                    # Check crew availability (simplified)
                    if t < crew.get('availability_start', 0) or (t + job['duration']) > crew.get('availability_end', len(time_slots)):
                        continue
                        
                    var = model.NewBoolVar(f'job_{job["id"]}_crew_{crew["id"]}_time_{t}')
                    schedule[(job['id'], crew['id'], t)] = var
                    assignments.append(var)
            
            if assignments:
                model.Add(sum(assignments) == 1)
            else:
                logger.warning(f"Job {job['id']} cannot be assigned to any crew due to constraints.")
                return {"status": "infeasible", "reason": f"Job {job['id']} unassignable"}

        # 2. Crews can only do one job at a time
        for crew in crews:
            for t in range(len(time_slots)):
                active_jobs = []
                for job in jobs:
                    for start_time in range(max(0, t - job['duration'] + 1), t + 1):
                        if (job['id'], crew['id'], start_time) in schedule:
                            active_jobs.append(schedule[(job['id'], crew['id'], start_time)])
                
                if active_jobs:
                    model.Add(sum(active_jobs) <= 1)

        # Objective: Maximize priority, Minimize completion time
        objective_terms = []
        for (j_id, c_id, t), var in schedule.items():
            job = next(j for j in jobs if j['id'] == j_id)
            priority_weight = job.get('priority', 1) * 10
            time_penalty = t  # Prefer earlier times
            objective_terms.append(var * (priority_weight * 100 - time_penalty))
            
        model.Maximize(sum(objective_terms))

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            result = []
            for (j_id, c_id, t), var in schedule.items():
                if solver.Value(var) == 1:
                    result.append({
                        "job_id": j_id,
                        "crew_id": c_id,
                        "start_time": t,
                        "end_time": t + next(j['duration'] for j in jobs if j['id'] == j_id)
                    })
            return {"status": "optimal", "schedule": result}
        else:
            return {"status": "infeasible"}
