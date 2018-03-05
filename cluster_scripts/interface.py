import requests
import logging
import datetime
from marshmallow import Schema, fields, pprint, post_load, pre_load





class OfferClass():

    def __init__(self, mem, gpus, cpus):
        self.mem = mem
        self.gpus = gpus
        self.cpus = cpus

class OfferSchema(Schema):
    mem = fields.Float()
    gpus = fields.Integer()
    cpus = fields.Float()

    @post_load(pass_many=True)
    def make_load(self, data, many):
        return OfferClass(**data)



class AgentClass():

    def __init__(self, hostname=None, ip=None, id=None , offer=None):
        self.hostname = hostname
        self.ip = ip
        self.id = id
        self.offer = offer


class AgentSchema(Schema):
    hostname = fields.Str(allow_none=True)
    ip = fields.Str(allow_none=True)
    id = fields.Str(allow_none=True)
    offer = fields.List(fields.Nested(OfferSchema()),allow_none=True)

    @post_load(pass_many=True)
    def make_load(self, data, many):
        return AgentClass(**data)


class AgentsClass():

    def __init__(self, agents):
        self.agents = agents


class AgentsSchema(Schema):
    agents = fields.List(fields.Nested(AgentSchema()))

    @post_load(pass_many=True)
    def make_load(self, data, many):
        return AgentsClass(**data)



class PortClass():

    def __init__(self, container, host=None):
        self.container = container
        self.host = host


class PortSchema(Schema):
    container = fields.Integer()
    host = fields.Integer(allow_none=True)

    @post_load(pass_many=True)
    def make_load(self, data, many):
        return PortClass(**data)



class TaskClass():

    def __init__(self, name, cpus,  mem, gpus, ports=[], status=None, mesos_id=None, creation_date=None, id=None, agent=None, force_pull=False, interactive=False, mount_home=False, command=None, reserved=None, image=None, lifetime=None):
        self.name = name
        self.id = id
        self.image = image
        self.status = status
        self.mesos_id = mesos_id
        self.creation_date = creation_date
        self.cpus = cpus
        self.command = command
        self.lifetime = lifetime
        self.mem = mem
        self.gpus = gpus
        self.reserved = reserved
        self.mount_home = mount_home
        self.interactive = interactive
        self.ports = ports
        self.agent = agent

class TaskSchema(Schema):
    name = fields.Str()
    id = fields.Integer(allow_none=True)
    image = fields.Str(allow_none=True)
    status = fields.Str(allow_none=True)
    mesos_id = fields.Str(allow_none=True)
    creation_date = fields.Str(allow_none=True)
    mem = fields.Float()
    gpus = fields.Integer()
    cpus = fields.Float()
    command = fields.Str(allow_none=True)
    reserved = fields.Boolean(allow_none=True)
    force_pull = fields.Boolean()
    lifetime = fields.Integer()

    ports = fields.List(fields.Nested(PortSchema()), allow_none=True)
    agent = fields.Str(allow_none=True)

    @post_load(pass_many=True)
    def make_load(self, data, many):
        return TaskClass(**data)

class TaskCreateSchema(Schema):
    name = fields.Str()
    id = fields.Integer(allow_none=True)
    image = fields.Str()
    status = fields.Str(allow_none=True)
    mesos_id = fields.Str(allow_none=True)
    creation_date = fields.Str(allow_none=True)
    mem = fields.Float()
    gpus = fields.Integer()
    cpus = fields.Float()
    command = fields.Str()
    reserved = fields.Boolean()
    interactive = fields.Boolean()
    mount_home = fields.Boolean()
    ports = fields.List(fields.Integer(), allow_none=True)
    force_pull = fields.Boolean()
    lifetime = fields.Integer()

class TasksClass():

    def __init__(self, tasks):
        self.tasks = tasks


class TasksSchema(Schema):
    tasks = fields.List(fields.Nested(TaskSchema()))

    @post_load(pass_many=True)
    def make_load(self, data, many):
        return TasksClass(**data)

class KillClass():

    def __init__(self, mesos_id):
        self.mesos_id = mesos_id


class KillSchema(Schema):
    mesos_id = fields.Str()

    @post_load(pass_many=True)
    def make_load(self, data, many):
        return KillClass(**data)


class Interface():

    def __init__(self, url, port, username, api_key):
        self.log = logging.getLogger(self.__class__.__name__)
        self.url = url
        self.port = port
        self.api_key = api_key
        self.username = username
        self.session = requests.Session()

        # Test api key


    def get_user_information(self):
        endpoint = '/api/user'

        url = self.url + endpoint
        headers = { 'Authorization':str(self.api_key)}
        response = self.session.get(url,  headers=headers)
        print(headers)
        if(response.status_code != 200):
            self.log.error('Frame returned error')

        # print(response.json())


    def get_tasks(self):
        endpoint = '/api/tasks'

        url = self.url + endpoint
        headers = { 'Authorization':str(self.api_key)}
        response = self.session.get(url,  headers=headers)
        print(headers)
        if(response.status_code != 200):
            self.log.error('Frame returned error')

        else:
            schema = TasksSchema()
            result = schema.loads('{\"tasks\":' + response.text + '}')
            tasks = result.data
            return tasks

    def get_agents(self):
        endpoint = '/api/agents'

        url = self.url + endpoint
        headers = { 'Authorization':str(self.api_key)}
        response = self.session.get(url,  headers=headers)
        if(response.status_code != 200):
            self.log.error('Frame returned error')

        else:
            schema = AgentsSchema()
            result = schema.loads('{\"agents\":' + response.text + '}')
            agents = result.data
            return agents

    def get_task(self, taskid):
        endpoint = '/api/task/'+str(taskid)

        url = self.url + endpoint
        headers = { 'Authorization':str(self.api_key)}
        response = self.session.get(url,  headers=headers)
        # print(headers)
        if(response.status_code != 200):
            self.log.error('Frame returned error')

        # print(response.text)
        schema = TaskSchema()
        result = schema.loads(response.text)
        task = result.data
        # print(task)
        return task




    def create_task(self, name, image, cpus, gpus, mem, reserved, ports, force_pull, interactive, mount_home, cmd="", lifetime=24*60):
        endpoint = '/api/tasks'

        task = TaskClass(name=name, image=image, cpus=cpus, gpus=gpus, mem=mem, reserved=reserved, ports=ports, command=cmd, force_pull=force_pull, interactive=interactive, mount_home=mount_home, lifetime=lifetime)
        print(task)
        schema = TaskCreateSchema()
        result = schema.dump(task)
        data = result.data
        print(data)

        url = self.url + endpoint
        headers = { 'Accept': 'application/json','Content-Type': 'application/json', 'Authorization':str(self.api_key)}
        response = self.session.post(url,  headers=headers, json=data)

        if(response.status_code != 201):
            self.log.error('Frame returned error')
            print(response.json())


        schema = TaskSchema()
        result = schema.loads(response.text)
        task = result.data
        return task

    def kill_task(self, mesos_id):
        endpoint = '/api/killtask'
        task = KillClass(mesos_id=mesos_id)
        schema = KillSchema()
        result = schema.dump(task)
        data = result.data

        url = self.url + endpoint
        headers = { 'Accept': 'application/json','Content-Type': 'application/json', 'Authorization':str(self.api_key)}
        response = self.session.post(url,  headers=headers, json=data)

        if(response.status_code != 201):
            self.log.error('Frame returned error')


def main():

    iface = Interface(url="http://127.0.0.1",port=80, username="thomas.kurmann", api_key="498716f5-edac-445f-91f8-7abcb6c066db")

    iface.get_user_information()
    iface.get_tasks()
    iface.get_agents()
    iface.get_task(taskid=8)


if __name__ == "__main__":
    main()
