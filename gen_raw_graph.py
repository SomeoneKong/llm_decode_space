import asyncio
import math
from typing import Optional

from llm_client import LlmClient, TokenProbInfo
from pydantic import BaseModel
from threading import Lock


class GraphNodeModel(BaseModel):
    id: int = -1
    father_node_id: Optional[int]
    token_info: Optional[TokenProbInfo]
    eos: bool = False
    accum_prob: float


class GraphNode:
    def __init__(self, model: GraphNodeModel):
        self.model = model
        self.father = None

    def get_all_prefix_token_ids(self):
        prefix = []
        node = self
        while node is not None and node.model.token_info is not None:
            prefix.append(node.model.token_info.token_id)
            node = node.father
        prefix.reverse()
        return prefix

    def get_all_prefix_text(self):
        prefix = ''
        node = self
        while node is not None and node.model.token_info is not None:
            prefix = node.model.token_info.token + prefix
            node = node.father
        return prefix

    def __str__(self):
        if self.model.token_info is None:
            return f'{self.model.id} {self.model.accum_prob}'
        return f'{self.model.id} {self.model.accum_prob:.6f} [{repr(self.father.get_all_prefix_text())}] {math.exp(self.model.token_info.logprob):.6f} {self.model.token_info.token_id} {repr(self.model.token_info.token)} {self.model.token_info.eos}'


class Graph:
    def __init__(self):
        self.nodes = dict()
        self.update_lock = Lock()

        self.expand_queue = []
        self.expand_queue_sorted = False
        self.expand_queue_lock = Lock()

        self.closed_node_list = []

    def add_node(self, node_model: GraphNodeModel):
        node = GraphNode(node_model)
        if node_model.father_node_id is not None:
            node.father = self.nodes[node_model.father_node_id]
        if node_model.id == -1:
            with self.update_lock:
                node_model.id = len(self.nodes)
                self.nodes[node_model.id] = node
        else:
            self.nodes[node_model.id] = node

        if not node.model.eos:
            with self.expand_queue_lock:
                self.expand_queue.append(node)
                self.expand_queue_sorted = False
        else:
            self.closed_node_list.append(node)

    def sort_and_get_next_expand_node(self):
        with self.expand_queue_lock:
            if not self.expand_queue_sorted:
                self.expand_queue.sort(key=lambda x: x.model.accum_prob)
                self.expand_queue_sorted = True
            if len(self.expand_queue) == 0:
                return None
            return self.expand_queue.pop()


async def fill_graph(llm_client,
                     message_list,
                     temperature=0.8,
                     top_p=0.95,
                     max_query_num=1000,
                     parallel_job_num=20,
                     ):
    graph = Graph()
    start_node_model = GraphNodeModel(
        id=0,
        father_node_id=None,
        token_info=None,
        accum_prob=0,
    )
    graph.add_node(start_node_model)

    query_task_list = []

    async def expand_node(query_id, node):
        query_prefix_token_ids = node.get_all_prefix_token_ids()
        next_token_dist_result = await llm_client.calc_next_token_dist(
            message_list,
            query_prefix_token_ids,
            temperature=temperature,
            top_p=top_p
        )

        result_nodes = []
        for token_prob in next_token_dist_result['token_dist']:
            new_node_model = GraphNodeModel(
                father_node_id=node.model.id,
                token_info=token_prob,
                eos=token_prob.eos,
                accum_prob=node.model.accum_prob + token_prob.logprob,
            )
            result_nodes.append(new_node_model)

        return query_id, node, result_nodes

    query_num = 0
    while True:
        if len(query_task_list) == 0:
            done_task, pending_task = set(), set()
        else:
            done_task, pending_task = await asyncio.wait(query_task_list)

        if len(done_task) > 0:
            for task in done_task:
                query_id, job_node, result_nodes = task.result()
                for node_model in result_nodes:
                    graph.add_node(node_model)
                    print(f'{query_id} {len(graph.expand_queue)} {len(graph.closed_node_list)} | {job_node.model.id} gen node: {node_model.id} {node_model.accum_prob:.6f} [{repr(graph.nodes[node_model.father_node_id].get_all_prefix_text())}] {node_model.token_info.token_id} {repr(node_model.token_info.token)} {node_model.token_info.eos}')
                query_task_list.remove(task)

        query_task_list = list(pending_task)

        while len(query_task_list) < parallel_job_num and query_num < max_query_num:
            node = graph.sort_and_get_next_expand_node()
            if node is None:
                break
            query_task_list.append(asyncio.create_task(expand_node(query_num, node)))
            query_num += 1

        if len(query_task_list) == 0:
            break

        await asyncio.sleep(0.05)

    print(f'query_num: {query_num} {len(graph.nodes)} {len(graph.expand_queue)}')

    return graph

prompt = '''
下列事件中，属于必然事件的是__

A. 任意数的绝对值都是正数

B. 两直线被第三条直线所截，同位角相等

C. 如果a、b都是实数，那么a+b=b+a

D. 抛掷1个均匀的骰子，出现6点朝上
'''.strip()


def main():
    import asyncio

    client = LlmClient()
    message_list = [
        {"role": "user", "content": prompt},
    ]
    graph = asyncio.run(
        fill_graph(client, message_list, temperature=0.8, top_p=0.9, max_query_num=20000)
    )

    for node in graph.nodes.values():
        # print(node)
        pass


if __name__ == "__main__":
    main()
