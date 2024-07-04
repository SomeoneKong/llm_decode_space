import math

import openai
from typing import List
from pydantic import BaseModel


class TokenProbInfo(BaseModel):
    token_id: int
    token: str
    logprob: float
    eos: bool
    bytes: List[int]


class LlmClient:
    def __init__(self,
                 base_url="http://localhost:6006/v1",
                 api_key="xxxx",
                 model_name="glm-4-9b-chat",
                 ):
        self.client = openai.AsyncClient(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name

    def trim_prob_dist(self, token_dist: List[TokenProbInfo], top_p: float):
        token_dist = sorted(token_dist, key=lambda x: x.logprob, reverse=True)
        accum_prob = 0
        for i, token_prob in enumerate(token_dist):
            accum_prob += math.exp(token_prob.logprob)
            if accum_prob >= top_p:
                break
        if accum_prob < top_p:
            print(f'warning: accum_prob < top_p, token_num={len(token_dist)} accum_prob={accum_prob}, top_p={top_p}')
            return token_dist
        return token_dist[:i + 1]

    async def calc_next_token_dist(self,
                                   message_list,
                                   prefix: List[int],
                                   top_p: float = 0.9,
                                   temperature: float = 0.8,
                                   ):
        for i in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list,
                    temperature=temperature,
                    extra_body={
                        "force_answer_prefix_token_ids": prefix,
                        "output_log_prob_token_id": True,
                    },
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=100,
                )
                break
            except Exception as e:
                print(f'error on run {i}: {e}')

        choices0 = response.choices[0]
        # print(choices0.finish_reason)

        token_dist = choices0.logprobs.content[0].top_logprobs
        token_dist = [TokenProbInfo(
            token_id=token_prob.token_id,
            token=token_prob.token,
            logprob=token_prob.logprob,
            eos=token_prob.eos,
            bytes=token_prob.bytes,
        ) for token_prob in token_dist]
        trimmed_token_dist = self.trim_prob_dist(token_dist, top_p)
        # print(trimmed_token_dist)

        return {
            'token_dist': trimmed_token_dist,
            'all_token_dist': token_dist,
        }

    async def sample_trace_stream(self,
                                  message_list,
                                  prefix: List[int],
                                  prefix_logprob: float = 0,
                                  top_p: float = 0.9,
                                  temperature: float = 0.8,
                                  max_tokens=1000,
                                  ):
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=message_list,
            temperature=temperature,
            extra_body={
                "force_answer_prefix_token_ids": prefix,
                "output_log_prob_token_id": True,
            },
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=50,
            stream=True,
            timeout=20,
        )

        accum_prefix = prefix[:]
        accum_logprob = prefix_logprob
        async with response:
            async for message in response:
                choice0 = message.choices[0]
                if choice0.delta.content is None:
                    continue

                delta_token = choice0.delta.content
                logprobs_info = choice0.logprobs.content[0]
                delta_token_id = logprobs_info.token_id
                delta_token_logprob = logprobs_info.logprob
                top_token_dist = logprobs_info.top_logprobs

                token_dist = [TokenProbInfo(
                    token_id=token_prob.token_id,
                    token=token_prob.token,
                    logprob=token_prob.logprob,
                    eos=token_prob.eos,
                    bytes=token_prob.bytes,
                ) for token_prob in top_token_dist]
                trimmed_token_dist = self.trim_prob_dist(token_dist, top_p)

                delta_info = {
                    'prefix': accum_prefix[:],
                    'prefix_logprob': accum_logprob,

                    'token_dist': trimmed_token_dist,
                    'all_token_dist': token_dist,

                    'delta_token': delta_token,
                    'delta_token_id': delta_token_id,
                    'delta_token_logprob': delta_token_logprob,
                    'finished_reason': choice0.finish_reason,
                }

                out_of_top_p = delta_token_id not in [token_prob.token_id for token_prob in trimmed_token_dist]
                if out_of_top_p:
                    delta_info['finished_reason'] = 'out_of_top_p'

                yield delta_info

                if out_of_top_p:
                    break

                accum_prefix.append(delta_token_id)
                accum_logprob += delta_token_logprob


    async def sample_trace_to_end_stream(
            self,
            message_list,
            prefix: List[int],
            prefix_logprob: float = 0,
            top_p: float = 0.9,
            temperature: float = 0.8,
            max_tokens=1000,
            ):
        while True:
            delta_info = None
            async for delta_info in self.sample_trace_stream(
                message_list,
                prefix,
                prefix_logprob,
                top_p,
                temperature,
                max_tokens,
            ):
                if delta_info['finished_reason'] == 'out_of_top_p':
                    delta_token_logprob = delta_info['delta_token_logprob']
                    pre_sum_prob = sum([math.exp(token_prob.logprob) for token_prob in delta_info['token_dist'] if token_prob.logprob > delta_token_logprob])
                    # print(f'drop out_of_top_p: {repr(delta_info["delta_token"])} {delta_info["delta_token_id"]} {delta_info["delta_token_logprob"]:.6f} {pre_sum_prob}')
                    break
                yield delta_info

            if delta_info['finished_reason'] == 'stop':
                break

            # print(f'continue from {delta_info["finished_reason"]} ...')
            prefix = delta_info['prefix']
            prefix_logprob = delta_info['prefix_logprob']




if __name__ == "__main__":
    import asyncio

    client = LlmClient()

    prompt = '''
下列事件中，属于必然事件的是__

A. 任意数的绝对值都是正数

B. 两直线被第三条直线所截，同位角相等

C. 如果a、b都是实数，那么a+b=b+a

D. 抛掷1个均匀的骰子，出现6点朝上
'''.strip()

    message_list = [
        {"role": "user", "content": prompt},
    ]
    prefix = [
    ]

    async def test_main():
        async for delta in client.sample_trace_to_end_stream(message_list, prefix, max_tokens=1000):
            print(f'prefix={delta["prefix"][-5:]}, delta={delta["delta_token_id"]}, {repr(delta["delta_token"])}, {delta["delta_token_logprob"]:.6f}, {delta["finished_reason"]}, {delta["token_dist"]}')

    asyncio.run(
        test_main()
    )
