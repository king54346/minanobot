"""
nanobot.providers.langchain_provider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

基于 LangChain ``init_chat_model`` 的 LLM Provider 实现。

与 LiteLLMProvider 的定位对比
-----------------------------
- **LiteLLMProvider**：通过 `litellm` 库的 `acompletion` 一次性调用即完成
  多厂商路由，不引入 LangChain 生态的依赖。
- **LangChainProvider**：使用 LangChain 的 ``init_chat_model()`` 统一入口
  自动选择正确的 ChatModel 实现（ChatOpenAI / ChatAnthropic / ChatGoogle 等）。
  其好处是：
  1) **一个入口覆盖所有厂商** — 不需要硬编码 ChatOpenAI，只需传入
     ``model`` 和 ``model_provider`` 即可自动路由；
  2) 可以直接复用 LangChain 生态的各种中间件（回调、缓存、链等）；
  3) 新增厂商时只需安装对应的 ``langchain-xxx`` 包，无需改代码。

支持的厂商（取决于安装的 langchain 集成包）
--------------------------------------------
- ``openai``       → pip install langchain-openai
- ``anthropic``    → pip install langchain-anthropic
- ``google_genai`` → pip install langchain-google-genai
- ``deepseek``     → pip install langchain-deepseek
- ``ollama``       → pip install langchain-ollama
- ...更多见 https://python.langchain.com/docs/integrations/chat/

依赖
----
基础依赖（必装）::

    pip install langchain>=0.3.0

然后根据需要的厂商安装对应集成包（如上所示）。

``init_chat_model`` 简介
-------------------------
``langchain.chat_models.init_chat_model(model, model_provider, **kwargs)``
会根据 ``model_provider``（如 ``"openai"``）或从 ``model`` 名称中自动推断
厂商，然后实例化对应的 ``BaseChatModel`` 子类。这样就不需要在代码里
``if/elif`` 各种 ChatModel 类了。
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

# ---- 厂商名映射表 ----
# nanobot 的 provider registry 用的名称 → init_chat_model 接受的 model_provider 值。
# 如果用户传入的 model_provider 已经是 init_chat_model 可识别的名称，则无需映射。
# 此表仅用于 nanobot 体系内的"翻译"，可根据需要扩展。
_PROVIDER_ALIAS: dict[str, str] = {
    # nanobot registry name  →  langchain model_provider name
    "openai":      "openai",
    "anthropic":   "anthropic",
    "deepseek":    "deepseek",
    "google":      "google_genai",
    "gemini":      "google_genai",
    "ollama":      "ollama",
    "fireworks":   "fireworks",
    "together":    "together",
    "mistral":     "mistralai",
    "groq":        "groq",
    "bedrock":     "bedrock",
    "azure":       "azure_openai",
    "moonshot":    "openai",       # Moonshot 兼容 OpenAI 协议
    "dashscope":   "openai",       # 通义千问兼容 OpenAI 协议
    "zhipu":       "openai",       # 智谱兼容 OpenAI 协议
    "minimax":     "openai",       # MiniMax 兼容 OpenAI 协议
    "vllm":        "openai",       # vLLM 本地部署兼容 OpenAI 协议
    "openrouter":  "openai",       # OpenRouter 网关兼容 OpenAI 协议
    "aihubmix":    "openai",       # AiHubMix 网关兼容 OpenAI 协议
}


class LangChainProvider(LLMProvider):
    """
    基于 LangChain ``init_chat_model`` 的通用 LLM Provider 实现。

    典型用法::

        # 自动推断厂商（从模型名 "gpt-4o" 推断为 openai）
        provider = LangChainProvider(
            api_key="sk-xxx",
            default_model="gpt-4o",
        )

        # 显式指定厂商
        provider = LangChainProvider(
            api_key="sk-ant-xxx",
            default_model="claude-sonnet-4-20250514",
            model_provider="anthropic",
        )

        # 使用 OpenAI 兼容协议的第三方端点（OpenRouter / vLLM / 通义千问等）
        provider = LangChainProvider(
            api_key="sk-or-xxx",
            api_base="https://openrouter.ai/api/v1",
            default_model="anthropic/claude-sonnet-4-20250514",
            model_provider="openrouter",          # 会映射为 "openai" 协议
        )

        response = await provider.chat(messages=[{"role": "user", "content": "你好"}])

    Parameters
    ----------
    api_key : str | None
        模型服务的 API Key。
    api_base : str | None
        自定义 API 端点 URL。对于 OpenAI 兼容协议的第三方服务必填。
        原生厂商（如 Anthropic、Google）一般不需要。
    default_model : str
        当调用 ``chat()`` 时未显式传入 model 参数时所使用的默认模型名。
    model_provider : str | None
        厂商标识，传给 ``init_chat_model`` 的 ``model_provider`` 参数。
        - 传 None 时，``init_chat_model`` 会尝试从 model 名称自动推断。
        - 传 nanobot registry 名称（如 ``"openrouter"``）会自动翻译。
        - 传 LangChain 原生名称（如 ``"anthropic"``）直接使用。
    extra_headers : dict[str, str] | None
        附加到每次请求的自定义 HTTP 头（例如 AiHubMix 需要的 APP-Code）。
    configurable_fields : list[str] | None
        允许运行时动态切换的字段列表，默认 ``["model", "temperature", "max_tokens"]``。
        传入后 ``init_chat_model`` 会使用 ``configurable_fields`` 参数，
        使得 ``chat()`` 里可以通过 ``config`` 动态覆盖这些参数。
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o",
        model_provider: str | None = None,
        extra_headers: dict[str, str] | None = None,
        configurable_fields: list[str] | None = None,
    ):
        # 调用基类，保存 api_key / api_base
        super().__init__(api_key, api_base)

        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        # ---- 解析 model_provider ----
        # 如果用户传入的是 nanobot registry 里的名称（如 "openrouter"），
        # 需要翻译成 init_chat_model 能识别的名称（如 "openai"）。
        self._raw_provider = model_provider
        self._lc_provider = self._resolve_provider(model_provider)

        # ---- 延迟导入 ----
        # 只有在真正创建 Provider 实例时才检查 langchain 是否安装，
        # 避免在未使用本 Provider 的环境中触发 ImportError。
        try:
            from langchain.chat_models import init_chat_model
        except ImportError as exc:
            raise ImportError(
                "LangChainProvider 需要安装 langchain>=0.3.0 。\n"
                "请运行：pip install langchain\n"
                "然后根据厂商安装对应集成包，如：pip install langchain-openai"
            ) from exc

        # ---- 构建 init_chat_model 的关键字参数 ----
        # init_chat_model 会把 **kwargs 透传给底层 ChatModel 的构造函数。
        init_kwargs: dict[str, Any] = {
            "temperature": 0.7,     # 默认温度，chat() 会动态覆盖
            "max_tokens": 4096,     # 默认 max_tokens，chat() 会动态覆盖
        }

        # api_key 的参数名因厂商而异：
        #   - OpenAI 兼容: api_key
        #   - Anthropic:   anthropic_api_key  (但 init_chat_model 也接受 api_key)
        #   - Google:      google_api_key
        # init_chat_model 对 api_key 做了统一处理，直接传 api_key 即可。
        if api_key:
            init_kwargs["api_key"] = api_key

        # 自定义端点（OpenRouter / AiHubMix / vLLM / 通义千问等）
        # 对于 OpenAI 兼容协议的服务，参数名为 base_url。
        if api_base:
            init_kwargs["base_url"] = api_base

        # 自定义请求头
        if self.extra_headers:
            init_kwargs["default_headers"] = self.extra_headers

        # ---- 使用 configurable_fields 支持运行时动态切换参数 ----
        # init_chat_model 的 configurable_fields 参数允许标记某些字段为
        # "可配置的"，之后可以通过 .with_config() 在运行时动态切换。
        # 默认允许切换 model、temperature、max_tokens。
        self._configurable_fields = configurable_fields or [
            "model", "temperature", "max_tokens"
        ]

        # ---- 创建 ChatModel 实例 ----
        # init_chat_model 根据 model_provider 或 model 名称自动选择
        # 正确的 ChatModel 子类（ChatOpenAI / ChatAnthropic 等）。
        self._llm = init_chat_model(
            model=default_model,
            model_provider=self._lc_provider,       # 可以为 None，让 LangChain 自动推断
            configurable_fields=self._configurable_fields,
            **init_kwargs,
        )

        logger.debug(
            f"LangChainProvider initialized: model={default_model}, "
            f"provider={self._lc_provider or 'auto'}, "
            f"api_base={api_base or 'default'}"
        )

    # ------------------------------------------------------------------
    # 辅助：厂商名映射
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_provider(provider: str | None) -> str | None:
        """
        将 nanobot 体系内的厂商名翻译为 ``init_chat_model`` 可识别的名称。

        - 如果 provider 在 _PROVIDER_ALIAS 中有映射，返回映射后的值。
        - 如果 provider 不在映射表中，原样返回（让 init_chat_model 自己处理）。
        - 如果 provider 为 None，返回 None（由 init_chat_model 从 model 名推断）。
        """
        if provider is None:
            return None
        return _PROVIDER_ALIAS.get(provider.lower(), provider)

    # ------------------------------------------------------------------
    # 核心方法：chat()
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        发起一次对话请求，返回项目内统一的 LLMResponse。

        Parameters
        ----------
        messages : list[dict]
            对话消息列表，每条格式为 ``{"role": "user"/"assistant"/"system", "content": "..."}``。
        tools : list[dict] | None
            OpenAI 格式的工具定义列表。传入后 LangChain 会自动启用 function calling。
        model : str | None
            本次请求使用的模型名，为 None 时使用 ``self.default_model``。
        max_tokens : int
            最大生成 token 数。
        temperature : float
            采样温度。

        Returns
        -------
        LLMResponse
            包含 content（文本回复）、tool_calls（工具调用请求）、usage（用量统计）等。
        """

        # ---- 1. 通过 with_config 动态覆盖模型参数 ----
        # 因为 __init__ 中使用了 configurable_fields=["model", "temperature", "max_tokens"]，
        # 所以可以通过 with_config() 在运行时临时切换这些参数，而不影响 self._llm 的默认值。
        effective_model = model or self.default_model
        configurable_overrides: dict[str, Any] = {
            "model": effective_model,
            "temperature": temperature,
            "max_tokens": max(1, max_tokens),       # 至少为 1，与 LiteLLMProvider 一致
        }
        llm = self._llm.with_config(configurable=configurable_overrides)

        # ---- 2. 绑定工具（如果有） ----
        # LangChain 的 bind_tools() 会自动把 OpenAI 格式的工具定义
        # 转换为模型所需的 function_call / tool_choice 参数。
        # 注意：bind_tools 需要底层 ChatModel 支持工具调用。
        if tools:
            llm = llm.bind_tools(tools)

        # ---- 3. 转换消息格式 ----
        # nanobot 内部使用 OpenAI 风格的 dict 列表；
        # LangChain 需要 BaseMessage 子类实例。
        lc_messages = self._convert_messages(messages)

        # ---- 4. 调用模型 ----
        try:
            response = await llm.ainvoke(lc_messages)
            return self._parse_response(response)
        except Exception as e:
            # 与 LiteLLMProvider 保持一致：出错时把异常信息放进 content 返回，
            # 让上层 AgentLoop 可以优雅处理，而不是直接崩溃。
            logger.error(f"LangChain LLM call failed: {e}")
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )

    # ------------------------------------------------------------------
    # 消息格式转换
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list:
        """
        将 nanobot 内部的 OpenAI 风格消息列表转换为 LangChain BaseMessage 对象列表。

        映射关系：
        - ``{"role": "system", ...}``    → ``SystemMessage``
        - ``{"role": "user", ...}``      → ``HumanMessage``
        - ``{"role": "assistant", ...}`` → ``AIMessage``（可能携带 tool_calls）
        - ``{"role": "tool", ...}``      → ``ToolMessage``（工具执行结果）

        对于 assistant 消息中的 tool_calls，也会被转换为 LangChain 的
        tool_calls 结构，以便模型能正确理解多轮工具对话的上下文。
        """
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""

            if role == "system":
                lc_messages.append(SystemMessage(content=content))

            elif role == "user":
                lc_messages.append(HumanMessage(content=content))

            elif role == "assistant":
                # assistant 消息可能带有 tool_calls（上一轮模型要求调用工具）
                raw_tool_calls = msg.get("tool_calls", [])
                if raw_tool_calls:
                    # 转换为 LangChain 的 tool_calls 格式
                    lc_tool_calls = []
                    for tc in raw_tool_calls:
                        func = tc.get("function", {})
                        args = func.get("arguments", {})
                        # arguments 可能是 JSON 字符串，也可能已经是 dict
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                        lc_tool_calls.append({
                            "id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "args": args,
                        })
                    lc_messages.append(
                        AIMessage(content=content, tool_calls=lc_tool_calls)
                    )
                else:
                    lc_messages.append(AIMessage(content=content))

            elif role == "tool":
                # 工具执行结果：必须携带 tool_call_id 以便模型关联
                lc_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.get("tool_call_id", ""),
                    )
                )
            else:
                # 未知 role 作为 HumanMessage 处理（兼容性保底）
                lc_messages.append(HumanMessage(content=content))

        return lc_messages

    # ------------------------------------------------------------------
    # 响应解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response) -> LLMResponse:
        """
        将 LangChain 的 AIMessage 响应转换为 nanobot 标准的 LLMResponse。

        处理内容：
        1. **文本回复** → ``LLMResponse.content``
        2. **工具调用** → ``LLMResponse.tool_calls`` (list[ToolCallRequest])
        3. **Token 用量** → ``LLMResponse.usage`` (dict)
           - LangChain >= 0.2 的模型会在 ``response.usage_metadata`` 中返回用量信息。
        4. **结束原因** → ``LLMResponse.finish_reason``
           - 有工具调用时为 ``"tool_calls"``，否则为 ``"stop"``。
        """
        # ---- 提取文本 ----
        content = response.content if isinstance(response.content, str) else ""

        # ---- 提取工具调用 ----
        tool_calls: list[ToolCallRequest] = []
        raw_tool_calls = getattr(response, "tool_calls", None) or []
        for tc in raw_tool_calls:
            # LangChain tool_calls 格式: {"id": ..., "name": ..., "args": {...}}
            tool_calls.append(
                ToolCallRequest(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    arguments=tc.get("args", {}),
                )
            )

        # ---- 提取 Token 用量 ----
        usage: dict[str, int] = {}
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            # LangChain 的 usage_metadata 是 dict-like，字段名与 OpenAI 略有不同
            usage = {
                "prompt_tokens": usage_meta.get("input_tokens", 0),
                "completion_tokens": usage_meta.get("output_tokens", 0),
                "total_tokens": usage_meta.get("total_tokens", 0),
            }

        # ---- 确定结束原因 ----
        finish_reason = "tool_calls" if tool_calls else "stop"

        # ---- 提取推理内容（如支持的模型） ----
        # 某些模型（如 DeepSeek-R1）会在 additional_kwargs 中返回推理过程
        additional = getattr(response, "additional_kwargs", {}) or {}
        reasoning_content = additional.get("reasoning_content", None)

        return LLMResponse(
            content=content or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=reasoning_content,
        )

    # ------------------------------------------------------------------
    # 默认模型
    # ------------------------------------------------------------------

    def get_default_model(self) -> str:
        """返回该 Provider 实例的默认模型名。"""
        return self.default_model

