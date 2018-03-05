from interface import Interface
from configparser import ConfigParser
import logging
from tabulate import tabulate
import sys
import os
import time
import cson
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv):
    log = logging.getLogger(__name__)
    config = ConfigParser()
    try:
        config.read("user.ini")
        username = config.get('user', 'username')
        api_key = config.get('user', 'api_key')
        host =  config.get('user', 'host')
    except:
        log.error("Error reading user.ini file")
        exit()

    if(len(argv) < 2):
        log.error("Please enter a task file")
    else:

        try:
            iface = Interface(url=host,port=80, username=username, api_key=api_key)
            iface.get_user_information()
            log.info("Credentials Valid")
        except:
            log.error("Invalid credentials")
            exit()

        with open(sys.argv[1]) as data_file:
            batch_tasks = json.load(data_file)

        tasks = iface.get_tasks()

        for task in batch_tasks:

            name = task['name']
            image = task['image']
            cpus = task['cpus']
            gpus = task['gpus']
            mem = task['mem']
            ports = task['ports']
            reserved = task['reserved']
            force_pull = task['force_pull']
            interactive = task['interactive']
            mount_home = task['mount_home']
            cmd = task['cmd']

            active_tasks = []
            if(len(tasks.tasks) > 0):
                for task in tasks.tasks:
                    if(task.status == "TASK_RUNNING"):
                        active_tasks.append(task)
                    elif(task.status == "TASK_QUEUED"):
                        active_tasks.append(task)


            tasks_with_that_name = []
            for task in active_tasks:
                print(task)
                if(task.name == name):
                    # log.info("Task exists")
                    tasks_with_that_name.append(task)

            cluster_task = ""
            if(len(tasks_with_that_name) == 0):
                # create job
                log.info("No existing task exists, creating")
                cluster_task = iface.create_task(name=name, image=image, cpus=cpus, gpus=gpus, mem=mem, ports=ports, reserved=reserved, force_pull=force_pull, interactive=interactive, mount_home=mount_home, cmd=cmd)
            elif(len(tasks_with_that_name) == 1):
                # one job exists, get status
                log.info("Found one task with that name")
                cluster_task = tasks_with_that_name[0]

            else:
                log.error("Found more than one task with that name!")


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
