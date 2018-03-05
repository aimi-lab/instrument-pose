from interface import Interface
from configparser import ConfigParser
import logging
from tabulate import tabulate
import sys
import os
import time
import cson

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_port_open(ip, port, raise_err=False):
    import socket
    from contextlib import closing

    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(2)
            result = sock.connect_ex((ip, port))
            return (True if (result == 0) else False)
    except socket.error as e:
        if raise_err:
            raise e
        return False



def connect_to_server(username, host,port):
    os.system('ssh ' + username + '@' + host + ' -p ' + port + " -o \"UserKnownHostsFile /dev/null\" -o \"LogLevel ERROR\" ")


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

        config_file = sys.argv[1]
        config.read(config_file)
        gpus = config.getint('job', 'gpus')
        cpus = config.getint('job', 'cpus')
        mem = config.getint('job', 'mem')
        ports= config.get('job', 'ports')
        cmd = config.get('job', 'command')
        if(ports != ""):
            ports = [x.strip() for x in ports.split(',')]
            ports = [int(i) for i in ports]
        image = config.get('job', 'image')
        name = config.get('job', 'name')
        attributes = config.get('job', 'attributes')
        attributes = [x.strip() for x in attributes.split(':')]
        attributes = [str(i) for i in attributes]
        attributes = dict(zip(attributes[0::2], attributes[1::2]))
        reserved = config.getboolean('job', 'reserved')
        force_pull = config.getboolean('job', 'force_pull')
        interactive = config.getboolean('job', 'interactive')
        mount_home = config.getboolean('job', 'mount_home')

        try:
            iface = Interface(url=host,port=80, username=username, api_key=api_key)
            iface.get_user_information()
            log.info("Credentials Valid")
        except:
            log.error("Invalid credentials")
            exit()

        tasks = iface.get_tasks()
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
            if(not cmd):
                log.info("No command given, you will be logged in")
                cluster_task = iface.create_task(name=name, image=image, cpus=cpus, gpus=gpus, mem=mem, ports=ports, reserved=reserved, force_pull=force_pull, interactive=interactive, mount_home=mount_home)
            else:
                log.info("Running command:")
                print(cmd)
                cluster_task = iface.create_task(name=name, image=image, cpus=cpus, gpus=gpus, mem=mem, ports=ports, reserved=reserved, force_pull=force_pull, interactive=interactive, mount_home=mount_home,cmd=cmd)
        elif(len(tasks_with_that_name) == 1):
            # one job exists, get status
            log.info("Found one task with that name")
            cluster_task = tasks_with_that_name[0]

        else:
            log.error("Found more than one task with that name!")

        cluster_task = iface.get_task(taskid=cluster_task.id)
        log.info("Waiting for Task to run")
        while(cluster_task.status != "TASK_RUNNING"): # tbd: change to enum not string
            cluster_task = iface.get_task(taskid=cluster_task.id)
            # print(cluster_task.status)
            time.sleep(1)

        log.info("Task running, connecting")
        # find port 22 and get ip of agent
        host_port = None

        if(len(cluster_task.ports) == 0):
            log.info("No SSH port spescified, exiting this application (your task is probably running non-interactive")

        for port in cluster_task.ports:
            if(port.container == 22):
                host_port = port.host

        agents = iface.get_agents()
        task_agent = None
        for agent in agents.agents:
            if(agent.id == cluster_task.agent):
                task_agent = agent

        for t in range(20):
            if(is_port_open(task_agent.ip, host_port, raise_err=False)):
                break;
            else:
                time.sleep(1)


        connect_to_server(username, task_agent.ip , str(host_port))
        #connect_to_server(username, task_agent.ip , "22")
        response = input('Do you wish to remove the job? [y/n]')
        if response == 'y':
            iface.kill_task(cluster_task.mesos_id)


if __name__ == "__main__":
    argv = sys.argv
    main(argv)
