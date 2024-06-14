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
            print(f'warning: accum_prob < top_p, accum_prob={accum_prob}, top_p={top_p}')
            return token_dist
        return token_dist[:i + 1]

    async def calc_next_token_dist(self,
                                   message_list,
                                   prefix: List[int],
                                   top_p: float = 0.9,
                                   temperature: float = 0.8,
                                   ):
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
            top_logprobs=20,
        )

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


if __name__ == "__main__":
    import asyncio

    client = LlmClient()
    message_list = [
        {"role": "user", "content": "你好"},
    ]
    prefix = [
    ]

    asyncio.run(
        client.calc_next_token_dist(message_list, prefix)
    )
