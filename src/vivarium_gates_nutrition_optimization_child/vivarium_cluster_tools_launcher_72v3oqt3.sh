
    export VIVARIUM_LOGGING_DIRECTORY=/mnt/team/simulation_science/pub/models/vivarium_gates_nutrition_optimization_child/results/model14.1/nigeria/2024_06_01_12_28_31/logs/2024_06_01_12_28_31_run/worker_logs
    export PYTHONPATH=/mnt/team/simulation_science/pub/models/vivarium_gates_nutrition_optimization_child/results/model14.1/nigeria/2024_06_01_12_28_31:$PYTHONPATH

    /ihme/homes/lutzes/.conda/envs/child_runs_v4/bin/rq worker -c settings         --name ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}         --burst         -w "vivarium_cluster_tools.psimulate.worker.core._ResilientWorker"         --exception-handler "vivarium_cluster_tools.psimulate.worker.core._retry_handler" vivarium

    