import asyncio
import math
from typing import Optional, List
from pydantic import BaseModel
from queue import PriorityQueue, Empty
from threading import Lock
from dataclasses import dataclass, field

from llm_client import LlmClient, TokenProbInfo
import openai


class GraphNodeModel(BaseModel):
    id: int = -1
    father_node_id: Optional[int]
    token_info: Optional[TokenProbInfo]
    eos: bool
    accum_logprob: float
    seq_token_num: int


class GraphNode:
    def __init__(self, model: GraphNodeModel):
        self.model = model
        self.father = None
        self.children = []

        self.total_trim_logprob = None
        self.accum_total_trim_logprob = None

    def trimmed_accum_logprob(self):
        result = self.model.accum_logprob - (self.father.accum_total_trim_logprob if self.father is not None else 0)
        assert result <= 0 + 1e-6, f'{result} {self.model.accum_logprob} {self.father.accum_total_trim_logprob if self.father is not None else 0}'
        return result

    def expand_rank_score(self):
        return self.trimmed_accum_logprob()

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
            return f'{self.model.id} {self.model.accum_logprob} {self.total_trim_logprob} {self.accum_total_trim_logprob}'
        return f'{self.model.id} {self.model.accum_logprob:.6f} {self.total_trim_logprob} {self.accum_total_trim_logprob} [{repr(self.father.get_all_prefix_text())}] {math.exp(self.model.token_info.logprob):.6f} {self.model.token_info.token_id} {repr(self.model.token_info.token)} {self.model.token_info.eos}'


class Graph:
    @dataclass(order=True)
    class PrioritizedItem:
        priority: float
        item: GraphNode = field(compare=False)

    def __init__(self):
        self.update_lock = Lock()

        self.nodes = dict()
        self.expand_queue = PriorityQueue()
        self.closed_node_list = []

    def add_nodes(self,
                  father_id: Optional[int],
                  node_model_list: List[GraphNodeModel],
                  node_list_to_expand: List[GraphNodeModel]
                  ):

        if father_id is not None:
            father_node = self.nodes[father_id]
        else:
            father_node = None

        node_list = []
        total_prob = 0
        for node_model in node_model_list:
            node = GraphNode(node_model)
            node.father = father_node
            if father_node is not None:
                father_node.children.append(node)
            node_list.append(node)

            total_prob += math.exp(node_model.token_info.logprob if node_model.token_info is not None else 0)

        expand_node_list = []
        closed_node_list = []
        for node in node_list:
            if node.model.eos:
                closed_node_list.append(node)
            if node.model in node_list_to_expand and not node.model.eos:
                expand_node_list.append(node)

        with self.update_lock:
            for node in node_list:
                if node.model.id == -1:
                    node.model.id = len(self.nodes)
                self.nodes[node.model.id] = node

            for node in closed_node_list:
                self.closed_node_list.append(node)

            if father_node is not None:
                father_node.total_trim_logprob = math.log(total_prob)
                father_node.accum_total_trim_logprob = math.log(total_prob)
                if father_node.father is not None:
                    father_node.accum_total_trim_logprob += father_node.father.accum_total_trim_logprob

            for node in expand_node_list:
                self.expand_queue.put(Graph.PrioritizedItem(priority= -node.expand_rank_score(), item=node))

        # print(f'add_nodes: {len(node_list)} {len(expand_node_list)} {len(closed_node_list)}')

    def add_exist_node_to_expand_queue(self, node_model: GraphNodeModel):
        node = self.nodes[node_model.id]
        self.expand_queue.put(Graph.PrioritizedItem(priority= -node.expand_rank_score(), item=node))


    def sort_and_get_next_expand_node(self):
        try:
            item = self.expand_queue.get_nowait()
        except Empty:
            return None
        return item.item


async def fill_graph(llm_client,
                     message_list,
                     temperature=0.8,
                     top_p=0.95,

                     sample_max_tokens=1000,
                     max_query_num=1000,
                     parallel_job_num=20,
                     ):
    graph = Graph()
    start_node_model = GraphNodeModel(
        id=0,
        father_node_id=None,
        token_info=None,
        eos=False,
        accum_logprob=0,
        seq_token_num=0,
    )
    graph.add_nodes(None, [start_node_model], [start_node_model])

    query_task_list = []

    async def expand_node(query_id, node, graph: Graph):
        current_node_model = node.model

        try:
            query_prefix_token_ids = node.get_all_prefix_token_ids()
            response_stream = llm_client.sample_trace_to_end_stream(
                message_list,
                query_prefix_token_ids,
                prefix_logprob=node.model.accum_logprob,
                temperature=temperature,
                top_p=top_p,
                max_tokens=sample_max_tokens,
            )

            async for delta_info in response_stream:
                finished_reason = delta_info['finished_reason']
                token_dist = delta_info['token_dist']

                if finished_reason is None:
                    delta_token_id = delta_info['delta_token_id']
                else:
                    delta_token_id = None

                next_node_model = None
                result_nodes = []
                expand_nodes = []
                for token_prob in token_dist:
                    new_node_model = GraphNodeModel(
                        father_node_id=current_node_model.id,
                        token_info=token_prob,
                        eos=token_prob.eos,
                        accum_logprob=current_node_model.accum_logprob + token_prob.logprob,
                        seq_token_num=current_node_model.seq_token_num + 1,
                    )
                    result_nodes.append(new_node_model)
                    if token_prob.token_id != delta_token_id:
                        expand_nodes.append(new_node_model)
                    else:
                        next_node_model = new_node_model

                graph.add_nodes(current_node_model.id, result_nodes, expand_nodes)
                # print(f'{len(graph.nodes)} {len(graph.expand_queue)} {len(graph.closed_node_list)} | {current_node_model.id} gen node: seq_len={current_node_model.seq_token_num} [{repr(graph.nodes[current_node_model.id].get_all_prefix_text())}] token_dist={token_dist}')

                current_node_model = next_node_model
                assert current_node_model is None or current_node_model.id != - 1
                if current_node_model is None:
                    break
        except openai.APIConnectionError as e:
            graph.add_exist_node_to_expand_queue(current_node_model)
            print(f'query {query_id} failed: {e}, add left node {current_node_model.id} [{current_node_model.token_info}] to queue')

    query_num = 0
    while True:
        if len(query_task_list) == 0:
            done_task, pending_task = set(), set()
        else:
            done_task, pending_task = await asyncio.wait(query_task_list, timeout=3, return_when=asyncio.FIRST_COMPLETED)

        if len(done_task) > 0:
            for task in done_task:
                query_task_list.remove(task)

        query_task_list = list(pending_task)

        while len(query_task_list) < parallel_job_num and query_num < max_query_num:
            node = graph.sort_and_get_next_expand_node()
            if node is None:
                break
            query_task_list.append(asyncio.create_task(expand_node(query_num, node, graph)))
            query_num += 1
            if node.father is not None:
                print(f'create query {query_num} query_task_list={len(query_task_list)} {len(graph.nodes)} {graph.expand_queue.qsize()}, {node.expand_rank_score():.6f} seq_len={node.model.seq_token_num} [{repr(graph.nodes[node.model.father_node_id].get_all_prefix_text())}] {repr(node.model.token_info.token)} {node.model.token_info.logprob:.6f}')

            # 每次只添加一个，然后等下一轮（3s）
            break

        if len(query_task_list) == 0:
            break

        print(f'nodes={len(graph.nodes)} queue={graph.expand_queue.qsize()} closed={len(graph.closed_node_list)} running={len(query_task_list)}')

    print(f'query_num: {query_num} {len(graph.nodes)} {graph.expand_queue.qsize()}')

    return graph

prompt = '''
下列事件中，属于必然事件的是__

A. 任意数的绝对值都是正数

B. 两直线被第三条直线所截，同位角相等

C. 如果a、b都是实数，那么a+b=b+a

D. 抛掷1个均匀的骰子，出现6点朝上

'''.strip()

prompt = 'python中是否有优先级队列？'

def main():
    import asyncio

    client = LlmClient()
    message_list = [
        {"role": "user", "content": prompt},
    ]
    graph = asyncio.run(
        fill_graph(client, message_list, temperature=0.8, top_p=0.9, max_query_num=1000, parallel_job_num=20)
    )

    for node in graph.nodes.values():
        # print(node)
        pass

    graph.closed_node_list.sort(key=lambda x: x.trimmed_accum_logprob(), reverse=True)
    for node in graph.closed_node_list:
        print(f'{node.trimmed_accum_logprob():.6f} {node.model.seq_token_num} [{repr(node.get_all_prefix_text())}]')

    # 为了优化数值稳定性，采用从小加到大的方式计算
    finished_prob = 0
    graph.closed_node_list.sort(key=lambda x: x.trimmed_accum_logprob())
    for node in graph.closed_node_list:
        finished_prob += math.exp(node.trimmed_accum_logprob())

    print(f'finished_prob: {finished_prob}')


if __name__ == "__main__":
    main()
