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
    agents = iface.get_agents()

    resource_list = []
    for agent in agents.agents:
        tot_cpu = 0
        tot_gpu = 0
        tot_mem = 0
        for offer in agent.offer:
            tot_cpu += offer.cpus
            tot_gpu += offer.gpus
            tot_mem += offer.mem

        resource_list.append([agent.hostname, tot_cpu, tot_gpu, tot_mem])


    print(tabulate(resource_list, headers=['Agent', 'CPUs', 'GPUs', 'Memory'], tablefmt='orgtbl'))


if __name__ == "__main__":
    main()
