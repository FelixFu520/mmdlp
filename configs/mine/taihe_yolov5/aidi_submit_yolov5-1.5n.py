import time
from aidisdk import AIDIClient, get_docker_image
from aidisdk.compute.dag import DagJobCustomConfig, DagStatus, JobHook
from aidisdk.compute.job_abstract import (
    GitCodeItem,
    RunningResourceConfig,
    StartUpConfig,
    MountItem, JobMountType, MountMode,
)
from aidisdk.compute.package_abstract import (
    CodePackageConfig,
    LocalPackageItem,
)

client = AIDIClient()
cpu_queue = "share-cpu-bcloud-beijing"
#gpu_queue = "project-4090-robot-algorithm-bcloud-beijing"
gpu_queue = "project-4090-robot-algorithm-idc-newage"

taskname = "fafu_yolov5_15n_"
taskname += "_%s" % str(time.time()).replace(".", "")[:10]
dag = client.dag.new_dag(
    name="train_%s" % taskname,
    project_id="GA2020005",
    queue_name=gpu_queue,
    code_package=CodePackageConfig(
            raw_package=LocalPackageItem(
                lpath="/home/users/fa.fu/taihe/", encrypt_passwd="123765", follow_softlink=False
            ),
            # git_packages=[],
        ),
    desc="train %s dag" % taskname,
    priority=5,
)
print("new_dag done.")

train = dag.new_job(
    name="train_%s" % taskname,
    job_type="train",
    startup=StartUpConfig(
        command="bash train_yolov5_1.5n.sh",
	#command="sleep infinity",
        startup_dir="/running_package/code_package"
    ),
    queue_name=gpu_queue,
    running_resource=RunningResourceConfig(
        #docker_image="docker.hobot.cc/dlp/hat:runtime-py3.8-torch2.0.1-cu118-2.2.1",  # noqa
	docker_image="docker.hobot.cc/aiot/hat:runtime-py3.8-torch1.13.0-cu116-1.4.1-yoloworld-numpy-python",
        instance=1,
        cpu=12,
        gpu=8,
        cpu_mem_ratio=8,
        walltime=20000,
    ),
    mount=[MountItem(JobMountType.BUCKET, "AIoT-data-bucket", MountMode.READ_AND_WRITE),],
    priority=5,
    schedule_type=None,
    max_retries=0,
    support_async_run=False,
    desc="train %s job" % taskname,
)
print("train job done.")


client.dag.submit_dag(dag, timeout=60)
client.dag.check_into_queue(dag=dag, timeout=60)
