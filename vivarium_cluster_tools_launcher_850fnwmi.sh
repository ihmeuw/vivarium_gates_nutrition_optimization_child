
    export VIVARIUM_LOGGING_DIRECTORY=/mnt/share/homes/sbachmei/scratch/child/ethiopia/2024_06_12_15_53_13/logs/2024_06_12_15_53_13_run/worker_logs
    export PYTHONPATH=/mnt/share/homes/sbachmei/scratch/child/ethiopia/2024_06_12_15_53_13:$PYTHONPATH

    /ihme/homes/sbachmei/miniconda3/envs/dev/bin/rq worker -c settings         --name ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}         --burst         -w "vivarium_cluster_tools.psimulate.worker.core._ResilientWorker"         --exception-handler "vivarium_cluster_tools.psimulate.worker.core._retry_handler" vivarium

    