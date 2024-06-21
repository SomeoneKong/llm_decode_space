# llm_decode_space

# 使用说明

需要配合vllm的定制版使用，项目地址：
https://github.com/SomeoneKong/vllm/tree/partial_mode

vllm服务启动参数示例：
```commandline
python -m vllm.entrypoints.openai.api_server --trust-remote-code --model model_path --served-model-name model_name --port port --max-logprobs 100
```

