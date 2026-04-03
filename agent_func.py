

#See: https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/python/agent_func_gem_multiturn.py
#Interface: line 141 - AgentExecutor definition expected
#Also define environment here

from openrlhf.utils.agent import MultiTurnAgentExecutor, AgentInstanceBase

class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        pass

    async def reset():
        pass

    async def step():
        pass



#OpenRLHF looks for this
#Definition: https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/agent.py#L31
class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)

