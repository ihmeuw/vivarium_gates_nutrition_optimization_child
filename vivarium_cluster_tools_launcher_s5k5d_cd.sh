
    export VIVARIUM_LOGGING_DIRECTORY=/mnt/team/simulation_science/pub/models/vivarium_gates_nutrition_optimization_child/results/model13.0/ethiopia/2024_04_11_09_35_50/logs/2024_04_11_09_35_50_run/worker_logs
    export PYTHONPATH=/mnt/team/simulation_science/pub/models/vivarium_gates_nutrition_optimization_child/results/model13.0/ethiopia/2024_04_11_09_35_50:$PYTHONPATH

    /ihme/homes/lutzes/.conda/envs/child_run_v2/bin/rq worker -c settings         --name ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}         --burst         -w "vivarium_cluster_tools.psimulate.worker.core._ResilientWorker"         --exception-handler "vivarium_cluster_tools.psimulate.worker.core._retry_handler" vivarium

    