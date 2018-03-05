from interface import Interface
from configparser import ConfigParser
import logging
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def main():
    log = logging.getLogger(__name__)
    config = ConfigParser()
    config.read("user.ini")

    username = config.get('user', 'username')
    api_key = config.get('user', 'api_key')
    host =  config.get('user', 'host')
    try:
        iface = Interface(url=host,port=80, username=username, api_key=api_key)
    except:
        log.error("Invalid credentials")

    log.info("Connection established")
    tasks = iface.get_tasks()

    running_tasks = []
    queued_tasks = []
    for task in tasks.tasks:
        if(task.status == "TASK_RUNNING"):
            port_str = ""
            for port in task.ports:
                port_str += str(port.container) + ":" + str(port.host) + ","

            running_tasks.append([task.name, task.cpus, task.gpus, task.mem, port_str])
        elif(task.status == "TASK_QUEUED"):
            queued_tasks.append([task.name, task.cpus, task.gpus, task.mem])

    print("RUNNING TASKS:")
    print(tabulate(running_tasks, headers=['Agent', 'CPUs', 'GPUs', 'Memory', 'Ports'], tablefmt='orgtbl'))
    print("QUEUED TASKS:")
    print(tabulate(queued_tasks, headers=['Agent', 'CPUs', 'GPUs', 'Memory'], tablefmt='orgtbl'))

if __name__ == "__main__":
    main()
